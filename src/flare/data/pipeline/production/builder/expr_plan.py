"""ExprPlan — aggregates FeaCross + Encode + FidCross into a single lazy PyArrow Acero plan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.encoding.base import (
    pack_sparse_i64_low_bits,
)
from flare.data.pipeline.production.builder.encoding.protocol import encoded_missing_for
from flare.data.schema.feature_catalog import (
    FeatureOutputKind,
    FeatureRepresentation,
    storage_column_name,
)


def _enc_output_name(slot: int, output_kind: FeatureOutputKind) -> str:
    rep = (
        FeatureRepresentation.DENSE
        if output_kind is FeatureOutputKind.DENSE_FLOAT
        else FeatureRepresentation.FID
    )
    return storage_column_name(slot, rep)


if TYPE_CHECKING:
    from flare.data.pipeline.production.builder.cross.protocol import (
        FeaCrossExpr,
        FidCrossExpr,
    )
    from flare.data.pipeline.production.builder.encoding.protocol import EncoderExpr


class ExprPlan:
    """Aggregates all cross and encoding expressions into a PyArrow Acero plan.

    Execution order (non-commutative):
      1. fea_cross: produce new raw feature columns (sub/div/hitcount)
      2. encode: all raw features → fid/float (including step 1 new columns)
      3. fid_cross: produce new fid columns from encoded fid columns (bitpack/cartesian)
      4. post-fid encode: produce final GPU-ready category indices
    """

    __slots__ = (
        "_encode_specs",
        "_fea_cross_ops",
        "_fid_cross_ops",
        "_input_cols",
        "_output_cols",
        "_output_missing_values",
        "_post_fid_encode_specs",
        "_simd_bitpack_specs",
        "_sparse_output_columns",
    )

    def __init__(
        self,
        fea_cross_ops: Sequence[FeaCrossExpr],
        encode_specs: list[tuple[str, int, int, EncoderExpr]],
        fid_cross_ops: Sequence[FidCrossExpr],
        output_cols: list[str],
        *,
        post_fid_encode_specs: Sequence[tuple[str, int, int, EncoderExpr]] = (),
    ) -> None:
        """Initialize the expression plan.

        Args:
            fea_cross_ops: Pre-encode cross operators.
            encode_specs: List of (col_name, slot, length, encoder) tuples.
            fid_cross_ops: Post-encode cross operators.
            output_cols: Final output column names to select.
            post_fid_encode_specs: Encoders applied after FID crosses.
        """
        self._fea_cross_ops = fea_cross_ops
        self._fid_cross_ops = fid_cross_ops
        self._post_fid_encode_specs = list(post_fid_encode_specs)
        self._output_cols = output_cols

        output_missing_values: dict[str, int | float] = {}
        sparse_output_columns: dict[str, None] = {}
        for _, slot, _, enc in encode_specs:
            output_name = _enc_output_name(slot, enc.output_kind)
            output_missing_values[output_name] = enc.encoded_missing(slot)
            if enc.output_kind is FeatureOutputKind.SPARSE_I64_FIDS:
                sparse_output_columns[output_name] = None
            else:
                sparse_output_columns.pop(output_name, None)
        for op in fid_cross_ops:
            output_missing_values[op.name] = encoded_missing_for(
                FeatureOutputKind.SPARSE_I64_FIDS,
                op.slot,
            )
        for _, slot, _, enc in post_fid_encode_specs:
            output_name = _enc_output_name(slot, enc.output_kind)
            output_missing_values[output_name] = enc.encoded_missing(slot)
            if enc.output_kind is FeatureOutputKind.SPARSE_I64_FIDS:
                sparse_output_columns[output_name] = None
            else:
                sparse_output_columns.pop(output_name, None)
        self._output_missing_values = output_missing_values
        self._sparse_output_columns = sparse_output_columns

        self._simd_bitpack_specs = []
        self._encode_specs = list(encode_specs)

        # Input projection: base cols MINUS raw
        base_input_cols = _compute_input_cols(
            fea_cross_ops, encode_specs, fid_cross_ops, output_cols
        )
        self._input_cols = base_input_cols

    def transform(self, batch: pa.Table) -> pa.Table:
        if self._simd_bitpack_specs:
            batch = _simd_bitpack_arrow(batch, self._simd_bitpack_specs)

        batch = _normalize_list_types(batch)

        df_cols = set(batch.column_names)
        missing_cols = [c for c in self._input_cols if c not in df_cols]
        if missing_cols:
            raise RuntimeError(
                f"Missing required columns in input batch: {missing_cols}"
            )

        context = _CrossExecutionContext(batch)

        for op in self._fea_cross_ops:
            arr = op.table_compute(context)
            if arr is None:
                raise RuntimeError(f"fea cross {op.name!r} produced no output")
            context.set_column(op.name, arr)

        for col_name, slot, length, enc in self._encode_specs:
            if not context.has_column(col_name):
                raise RuntimeError(
                    f"encoder input {col_name!r} for feature code {slot} is missing"
                )
            arr = enc.table_compute(context, col_name, slot, length)
            if arr is None:
                raise RuntimeError(
                    f"encoder for feature code {slot} produced no output"
                )
            out_name = _enc_output_name(slot, enc.output_kind)
            context.set_column(out_name, arr)

        for op in self._fid_cross_ops:
            arr = op.table_compute(context)
            if arr is None:
                raise RuntimeError(f"fid cross {op.name!r} produced no output")
            context.set_column(op.name, arr)

        for col_name, slot, length, enc in self._post_fid_encode_specs:
            if not context.has_column(col_name):
                raise RuntimeError(
                    f"post-fid encoder input {col_name!r} for feature code {slot} is missing"
                )
            arr = enc.table_compute(context, col_name, slot, length)
            if arr is None:
                raise RuntimeError(
                    f"post-fid encoder for feature code {slot} produced no output"
                )
            out_name = _enc_output_name(slot, enc.output_kind)
            context.set_column(out_name, arr)

        output_arrays = []
        for name in self._output_cols:
            values = context.column(name)
            if name in self._output_missing_values:
                values = _materialize_output_missing(
                    values,
                    self._output_missing_values[name],
                    column_name=name,
                    materialize_empty_list=name in self._sparse_output_columns,
                )
            else:
                _require_no_output_nulls(values, column_name=name)
            output_arrays.append(values)

        res = pa.Table.from_arrays(output_arrays, names=self._output_cols)

        return res


def _materialize_output_missing(
    values: pa.ChunkedArray,
    missing_value: float,
    *,
    column_name: str,
    materialize_empty_list: bool,
) -> pa.ChunkedArray:
    chunks: list[pa.Array] = []
    for chunk in values.chunks:
        if pa.types.is_list(chunk.type):
            flattened = chunk.flatten()
            if flattened.null_count:
                raise ValueError(
                    f"encoded list output {column_name!r} contains null values"
                )
            if materialize_empty_list:
                lengths = pc.list_value_length(chunk)
                missing = pc.or_(
                    pc.is_null(chunk),
                    pc.fill_null(pc.equal(lengths, 0), False),
                )
                if pc.any(missing).as_py():
                    chunk = pc.if_else(
                        missing,
                        pa.scalar([missing_value], type=chunk.type),
                        chunk,
                    )
                chunks.append(chunk)
                continue
            if chunk.null_count:
                raise ValueError(
                    f"encoded list output {column_name!r} contains null rows"
                )
            chunks.append(chunk)
            continue
        if chunk.null_count:
            chunk = pc.fill_null(
                chunk,
                pa.scalar(missing_value, type=chunk.type),
            )
        chunks.append(chunk)
    return pa.chunked_array(chunks, type=values.type)


def _require_no_output_nulls(
    values: pa.ChunkedArray,
    *,
    column_name: str,
) -> None:
    for chunk in values.chunks:
        if chunk.null_count:
            raise ValueError(f"output {column_name!r} contains null rows")
        if pa.types.is_list(chunk.type) and chunk.flatten().null_count:
            raise ValueError(f"output {column_name!r} contains null values")


class _CrossExecutionContext:
    __slots__ = ("_cast_cache", "_columns", "_flat_cache", "_offset_cache")

    def __init__(self, table: pa.Table) -> None:
        self._columns = {name: table.column(name) for name in table.column_names}
        self._cast_cache: dict[tuple[str, str], pa.ChunkedArray] = {}
        self._flat_cache: dict[tuple[str, str, int, str, bool], np.ndarray] = {}
        self._offset_cache: dict[tuple[str, str, int], np.ndarray] = {}

    def column(self, name: str) -> pa.ChunkedArray:
        return self._columns[name]

    def has_column(self, name: str) -> bool:
        return name in self._columns

    def set_column(self, name: str, values: pa.ChunkedArray) -> None:
        self._columns[name] = values
        self._cast_cache = {
            key: value for key, value in self._cast_cache.items() if key[0] != name
        }
        self._flat_cache = {
            key: value for key, value in self._flat_cache.items() if key[0] != name
        }
        self._offset_cache = {
            key: value for key, value in self._offset_cache.items() if key[0] != name
        }

    def cast(self, name: str, typ: pa.DataType) -> pa.ChunkedArray:
        key = (name, str(typ))
        cached = self._cast_cache.get(key)
        if cached is None:
            cached = self.column(name).cast(typ, safe=False)
            self._cast_cache[key] = cached
        return cached

    def flat_numpy(
        self,
        name: str,
        typ: pa.DataType,
        chunk_idx: int,
        np_type: type[np.generic],
        *,
        list_mode: bool,
    ) -> np.ndarray:
        key = (name, str(typ), chunk_idx, np.dtype(np_type).str, list_mode)
        cached = self._flat_cache.get(key)
        if cached is not None:
            return cached
        arr = self.cast(name, typ).chunks[chunk_idx]
        flat = arr.flatten() if list_mode else arr
        values = flat.to_numpy(zero_copy_only=False).astype(np_type)
        self._flat_cache[key] = values
        return values

    def offsets(self, name: str, typ: pa.DataType, chunk_idx: int) -> np.ndarray:
        key = (name, str(typ), chunk_idx)
        cached = self._offset_cache.get(key)
        if cached is not None:
            return cached
        arr = self.cast(name, typ).chunks[chunk_idx]
        values = np.asarray(arr.offsets).astype(np.int32)
        self._offset_cache[key] = values
        return values


def _normalize_list_types(table: pa.Table) -> pa.Table:
    new_fields = []
    has_changes = False

    for field in table.schema:
        t = field.type
        if pa.types.is_fixed_size_list(t) or pa.types.is_large_list(t):
            list_type = pa.list_(t.value_type)
            new_fields.append(pa.field(field.name, list_type, nullable=field.nullable))
            has_changes = True
        else:
            new_fields.append(field)

    if not has_changes:
        return table

    new_schema = pa.schema(new_fields)
    return table.cast(new_schema, safe=False)


def _simd_bitpack_arrow(
    table: pa.Table,
    bitpack_specs: list[tuple[str, int, int, int]],
) -> pa.Table:
    new_arrays = [table.column(i) for i in range(table.num_columns)]
    new_names = table.column_names.copy()

    for col_name, slot, length, bit_size in bitpack_specs:
        if length > 1:
            col = table.column(col_name)
            flat = col.flatten()
            flat_int64 = pc.cast(flat, pa.int64(), safe=False)
            packed_flat = pack_sparse_i64_low_bits(flat_int64, slot)

            if isinstance(col.type, pa.FixedSizeListType):
                fid_col = pa.FixedSizeListArray.from_arrays(packed_flat, length)
            else:
                shifted_offsets = pc.subtract(  # pyrefly: ignore[missing-attribute]
                    col.offsets, col.offsets[0]
                )
                fid_col = pa.ListArray.from_arrays(shifted_offsets, packed_flat)
        else:
            col = table.column(col_name)
            col_int64 = pc.cast(col, pa.int64(), safe=False)
            fid_col = pack_sparse_i64_low_bits(col_int64, slot)

        new_arrays.append(fid_col)
        new_names.append(storage_column_name(slot, FeatureRepresentation.FID))

    return pa.Table.from_arrays(new_arrays, names=new_names)


def _compute_input_cols(
    fea_cross_ops: Sequence[FeaCrossExpr],
    encode_specs: list[tuple[str, int, int, EncoderExpr]],
    fid_cross_ops: Sequence[FidCrossExpr],
    output_cols: list[str],
) -> list[str]:
    fea_cross_outputs = {op.name for op in fea_cross_ops}
    cols: set[str] = set()

    for op in fea_cross_ops:
        cols.update(op.dep_names)

    for col_name, _slot, _length, _enc in encode_specs:
        if col_name not in fea_cross_outputs:
            cols.add(col_name)

    encode_outputs = {
        _enc_output_name(slot, enc.output_kind) for _, slot, _, enc in encode_specs
    }

    fid_cross_outputs = {op.name for op in fid_cross_ops}
    produced = encode_outputs | fea_cross_outputs | fid_cross_outputs

    for op in fid_cross_ops:
        for dep in op.dep_names:
            if dep not in produced:
                cols.add(dep)

    for out in output_cols:
        if out not in produced:
            cols.add(out)
    return sorted(cols - produced)
