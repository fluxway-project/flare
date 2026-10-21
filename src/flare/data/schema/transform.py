from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.constants import NULL_FID_SENTINEL
from flare.data.schema.batch import RaggedFeatureBatch
from flare.data.schema.feature_catalog import FeatureOutputKind
from flare.data.schema.feature_layout import TrainingFeatureLayout


def _require_side_info_padding(
    field: pa.Field,
    *,
    expected_length: int,
    padding_value: int,
) -> pa.Scalar:
    if field.type != pa.list_(pa.int32()):
        raise ValueError(
            f"side-info feature {field.name!r} must use list<int32>, got {field.type}"
        )
    try:
        return pa.scalar(padding_value, type=pa.int32())
    except (ValueError, pa.ArrowInvalid) as exc:
        raise ValueError(
            f"list feature {field.name!r} has invalid int32 pad value"
        ) from exc


def _column_chunks(values: pa.Array | pa.ChunkedArray) -> tuple[pa.Array, ...]:
    if isinstance(values, pa.ChunkedArray):
        if values.num_chunks == 0:
            raise ValueError("training table column has no chunks")
        return tuple(values.chunks)
    return (values,)


def _concatenate_numpy_chunks(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("cannot concatenate an empty chunk sequence")
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _embedding_i32_indices(
    values: pa.Array,
    *,
    feature_name: str,
    null_index_sentinel: int,
) -> np.ndarray:
    if values.type != pa.int32():
        raise ValueError(
            f"side-info feature {feature_name!r} must use int32 values, got {values.type}"
        )
    if values.null_count:
        raise ValueError(f"side-info feature {feature_name!r} contains null values")
    array = values.to_numpy(zero_copy_only=False)
    invalid = (array < 0) & (array != null_index_sentinel)
    if invalid.any():
        raise ValueError(
            f"side-info feature {feature_name!r} contains negative dense indices"
        )
    return np.where(
        array == null_index_sentinel,
        0,
        array.astype(np.int64, copy=False) + 1,
    ).astype(np.int32, copy=False)


@numba.njit(nogil=True, cache=True)
def _copy_list_prefixes(offsets, values, valid_rows, output):
    width = output.shape[1]
    for row_index in range(output.shape[0]):
        if valid_rows[row_index]:
            start = offsets[row_index]
            length = min(offsets[row_index + 1] - start, width)
            for column_index in range(length):
                output[row_index, column_index] = values[start + column_index]


@numba.njit(nogil=True, cache=True)
def _encode_dense_scalar_indices(values, valid_rows, null_index_sentinel, output):
    for row_index in range(values.shape[0]):
        if valid_rows[row_index]:
            value = values[row_index]
            output[row_index] = 0 if value == null_index_sentinel else value + 1
        else:
            output[row_index] = 0


@numba.njit(nogil=True, cache=True)
def _encode_dense_list_indices(
    offsets,
    values,
    valid_values,
    valid_rows,
    width,
    null_index_sentinel,
    padding_value,
    output,
):
    encoded_padding = 0 if padding_value == null_index_sentinel else padding_value + 1
    output.fill(encoded_padding)
    for row_index in range(output.shape[0]):
        if not valid_rows[row_index]:
            continue
        start = offsets[row_index]
        length = min(offsets[row_index + 1] - start, width)
        for column_index in range(length):
            value_index = start + column_index
            if valid_values[value_index]:
                value = values[value_index]
                output[row_index, column_index] = (
                    0 if value == null_index_sentinel else value + 1
                )


@numba.njit(nogil=True, cache=True)
def _copy_sparse_feature_prefixes(output, starts, lengths, values, offsets):
    for row_index in range(lengths.shape[0]):
        destination_start = starts[row_index]
        source_start = offsets[row_index]
        for value_index in range(lengths[row_index]):
            output[destination_start + value_index] = values[source_start + value_index]


def _padded_list_values(
    column: pa.ListArray, width: int, padding: pa.Scalar
) -> pa.Array:
    if width <= 0:
        raise ValueError("side-info list width must be positive")
    values = column.values
    if values.null_count:
        values = pc.fill_null(values, padding)
    output = np.full((len(column), width), padding.as_py(), dtype=np.int32)
    _copy_list_prefixes(
        column.offsets.to_numpy(),
        values.to_numpy(),
        column.is_valid().to_numpy(zero_copy_only=False),
        output,
    )
    return pa.array(output.reshape(-1))


def _primitive_i32_buffer(values: pa.Array) -> np.ndarray:
    data_buffer = values.buffers()[1]
    if data_buffer is None:
        raise ValueError("int32 Arrow array has no data buffer")
    return np.frombuffer(
        data_buffer,
        dtype=np.int32,
        count=len(values),
        offset=values.offset * np.dtype(np.int32).itemsize,
    )


def _validate_dense_i32_values(
    values: np.ndarray,
    valid_values: np.ndarray,
    *,
    feature_name: str,
    null_index_sentinel: int,
) -> None:
    invalid = valid_values & (values < 0) & (values != null_index_sentinel)
    if invalid.any():
        raise ValueError(
            f"side-info feature {feature_name!r} contains negative dense indices"
        )


def _write_dense_scalar_indices(
    values: pa.Array,
    output: np.ndarray,
    *,
    feature_name: str,
    null_index_sentinel: int,
) -> None:
    if values.type != pa.int32():
        raise ValueError(
            f"side-info feature {feature_name!r} must use int32 values, got {values.type}"
        )
    raw_values = _primitive_i32_buffer(values)
    valid_values = values.is_valid().to_numpy(zero_copy_only=False)
    _validate_dense_i32_values(
        raw_values,
        valid_values,
        feature_name=feature_name,
        null_index_sentinel=null_index_sentinel,
    )
    output.fill(0)
    _encode_dense_scalar_indices(
        raw_values,
        valid_values,
        null_index_sentinel,
        output[:, 0],
    )


def _write_dense_list_indices(
    column: pa.ListArray,
    output: np.ndarray,
    *,
    feature_name: str,
    null_index_sentinel: int,
    padding_value: int,
) -> None:
    values = column.values
    if values.type != pa.int32():
        raise ValueError(
            f"side-info feature {feature_name!r} must use int32 values, got {values.type}"
        )
    raw_values = _primitive_i32_buffer(values)
    valid_values = values.is_valid().to_numpy(zero_copy_only=False)
    _validate_dense_i32_values(
        raw_values,
        valid_values,
        feature_name=feature_name,
        null_index_sentinel=null_index_sentinel,
    )
    offsets = column.offsets.to_numpy(zero_copy_only=False)
    output.fill(0)
    _encode_dense_list_indices(
        offsets,
        raw_values,
        valid_values,
        column.is_valid().to_numpy(zero_copy_only=False),
        output.shape[1],
        null_index_sentinel,
        padding_value,
        output,
    )


def _sparse_i64_values(values: pa.Array, *, feature_name: str) -> np.ndarray:
    if values.null_count:
        raise ValueError(f"sparse feature {feature_name!r} contains null IDs")
    if values.type == pa.int64():
        return values.to_numpy(zero_copy_only=False)
    raise ValueError(
        f"sparse feature {feature_name!r} must use int64, got {values.type}"
    )


def _system_i32_values(values: pa.Array, *, column_name: str) -> np.ndarray:
    chunks = []
    for chunk in _column_chunks(values):
        flat_values = chunk.flatten() if pa.types.is_list(chunk.type) else chunk
        if flat_values.type != pa.int32():
            raise ValueError(
                f"system column {column_name!r} must use int32, got {values.type}"
            )
        if flat_values.null_count:
            raise ValueError(f"system column {column_name!r} contains null values")
        chunks.append(flat_values.to_numpy(zero_copy_only=False))
    return _concatenate_numpy_chunks(chunks)


def _metadata_values(
    values: pa.Array,
    output_kind: FeatureOutputKind,
    *,
    column_name: str,
    null_index_sentinel: int,
) -> np.ndarray:
    output_chunks = []
    for chunk in _column_chunks(values):
        selected = chunk
        if pa.types.is_list(chunk.type):
            selected = pc.list_slice(
                chunk,
                0,
                1,
                return_fixed_size_list=True,
            ).values
        if output_kind is FeatureOutputKind.DENSE_I32_INDICES:
            output_chunks.append(
                _embedding_i32_indices(
                    selected,
                    feature_name=column_name,
                    null_index_sentinel=null_index_sentinel,
                )
            )
            continue
        if output_kind is FeatureOutputKind.SPARSE_I64_FIDS:
            output_chunks.append(_sparse_i64_values(selected, feature_name=column_name))
            continue
        if output_kind is FeatureOutputKind.DENSE_FLOAT:
            if selected.type != pa.float32():
                raise ValueError(
                    f"dense metadata {column_name!r} must use float32, got {selected.type}"
                )
            if selected.null_count:
                raise ValueError(f"dense metadata {column_name!r} contains null values")
            output_chunks.append(selected.to_numpy(zero_copy_only=False))
            continue
        raise ValueError(f"unsupported metadata output kind: {output_kind}")
    if output_chunks:
        return _concatenate_numpy_chunks(output_chunks)
    raise ValueError(f"unsupported metadata output kind: {output_kind}")


def _required_physical_columns(
    side_info_seq: Sequence[tuple[str, int]],
    categorical_feature_groups: Mapping[str, Sequence[str]],
    metadata_cols: Sequence[str],
    *,
    label_column: str,
    available_columns: Sequence[str],
    feature_layout: TrainingFeatureLayout,
) -> tuple[str, ...]:
    names: list[str] = []
    for feature_name, _length in side_info_seq:
        names.append(feature_layout.column_for(feature_name))
    for features in categorical_feature_groups.values():
        for feature_name in features:
            names.append(feature_layout.column_for(feature_name))
    for feature_name in metadata_cols:
        names.append(feature_layout.column_for(feature_name))
    names.append(
        label_column
        if label_column in available_columns
        else feature_layout.column_for(label_column)
    )
    return tuple(dict.fromkeys(names))


def table_to_training_dict(
    tbl: pa.Table,
    side_info_seq: Sequence[tuple[str, int]],
    categorical_feature_groups: Mapping[str, Sequence[str]],
    metadata_cols: Sequence[str],
    *,
    label_column: str,
    feature_layout: TrainingFeatureLayout,
) -> dict[str, Any]:
    row_count = tbl.num_rows
    res: dict[str, pa.Array | pa.ChunkedArray] = {}
    required_columns = _required_physical_columns(
        side_info_seq,
        categorical_feature_groups,
        metadata_cols,
        label_column=label_column,
        available_columns=tbl.column_names,
        feature_layout=feature_layout,
    )
    col_lookup = dict(zip(tbl.column_names, tbl.columns))
    for col_name in required_columns:
        if col_name not in col_lookup:
            continue
        col = col_lookup[col_name]
        if isinstance(col, pa.ChunkedArray) and col.num_chunks == 0:
            raise ValueError(f"training table column {col_name!r} has no chunks")
        res[col_name] = col

    out: dict[str, Any] = {}

    if side_info_seq:
        side_info = np.empty(
            (row_count, sum(length for _name, length in side_info_seq)),
            dtype=np.int32,
        )
        column_start = 0
        for fea, fl in side_info_seq:
            if (
                feature_layout.output_kind_for(fea)
                is not FeatureOutputKind.DENSE_I32_INDICES
            ):
                raise ValueError(
                    f"side-info feature {fea!r} is not a Dense int32 index"
                )
            col_name = feature_layout.column_for(fea)
            if col_name not in res:
                raise ValueError(
                    f"Feature not found in training batch: logical={fea!r} "
                    f"physical={col_name!r}; available={sorted(res)}"
                )
            col = res[col_name]
            if pa.types.is_list(col.type):
                _require_side_info_padding(
                    tbl.schema.field(col_name),
                    expected_length=fl,
                    padding_value=feature_layout.padding_for(fea),
                )
                destination_start = 0
                for source_chunk in _column_chunks(col):
                    destination_end = destination_start + len(source_chunk)
                    _write_dense_list_indices(
                        source_chunk,
                        side_info[
                            destination_start:destination_end,
                            column_start : column_start + fl,
                        ],
                        feature_name=col_name,
                        null_index_sentinel=feature_layout.null_index_sentinel,
                        padding_value=feature_layout.padding_for(fea),
                    )
                    destination_start = destination_end
            else:
                destination_start = 0
                for source_chunk in _column_chunks(col):
                    destination_end = destination_start + len(source_chunk)
                    _write_dense_scalar_indices(
                        source_chunk,
                        side_info[
                            destination_start:destination_end,
                            column_start : column_start + fl,
                        ],
                        feature_name=col_name,
                        null_index_sentinel=feature_layout.null_index_sentinel,
                    )
                    destination_start = destination_end
            column_start += fl
        out["side_info"] = side_info

    for gname, feas in categorical_feature_groups.items():
        if not feas:
            continue

        feature_chunks: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        total_lens = np.zeros(row_count, dtype=np.int32)
        feature_lengths = np.empty((row_count, len(feas)), dtype=np.int32)

        for feature_index, fea in enumerate(feas):
            if (
                feature_layout.output_kind_for(fea)
                is not FeatureOutputKind.SPARSE_I64_FIDS
            ):
                raise ValueError(f"sparse feature {fea!r} is not an int64 FID")
            col_name = feature_layout.column_for(fea)
            if col_name not in res:
                raise ValueError(
                    f"Feature not found in training batch: logical={fea!r} "
                    f"physical={col_name!r}; available={sorted(res)}"
                )
            col = res[col_name]
            chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            lens_parts: list[np.ndarray] = []
            for source_chunk in _column_chunks(col):
                if pa.types.is_list(source_chunk.type):
                    list_lengths = pc.list_value_length(source_chunk)
                    lens = list_lengths.to_numpy(zero_copy_only=False).astype(
                        np.int32, copy=False
                    )
                    offsets = source_chunk.offsets.to_numpy(zero_copy_only=False)
                    visible_values = source_chunk.values.slice(
                        int(offsets[0]), int(offsets[-1] - offsets[0])
                    )
                    vals = _sparse_i64_values(visible_values, feature_name=col_name)
                    offs = offsets[:-1] - offsets[0]
                else:
                    lens = np.ones(len(source_chunk), dtype=np.int32)
                    vals = _sparse_i64_values(source_chunk, feature_name=col_name)
                    offs = np.arange(len(source_chunk), dtype=np.int32)
                lens_parts.append(lens)
                chunks.append((lens, vals, offs))
            lens = _concatenate_numpy_chunks(lens_parts)
            total_lens += lens
            feature_lengths[:, feature_index] = lens
            feature_chunks.append(chunks)

        new_offsets = np.zeros(row_count + 1, dtype=np.int32)
        np.cumsum(total_lens, out=new_offsets[1:])

        new_vals = np.empty(new_offsets[-1], dtype=np.int64)
        current_starts = new_offsets[:-1].copy()

        for chunks in feature_chunks:
            row_start = 0
            for lens, vals, offs in chunks:
                row_end = row_start + len(lens)
                if len(vals) > 0:
                    _copy_sparse_feature_prefixes(
                        new_vals,
                        current_starts[row_start:row_end],
                        lens,
                        vals,
                        offs,
                    )
                current_starts[row_start:row_end] += lens
                row_start = row_end

        row_feature_offsets = np.arange(
            0,
            (row_count + 1) * len(feas),
            len(feas),
            dtype=np.int32,
        )
        flat_feature_lengths = feature_lengths.reshape(-1)
        feature_id_offsets = np.zeros(flat_feature_lengths.size + 1, dtype=np.int32)
        np.cumsum(flat_feature_lengths, out=feature_id_offsets[1:])
        out[gname] = RaggedFeatureBatch(
            flat_values=new_vals,
            nested_splits=(row_feature_offsets, feature_id_offsets),
            row_count=row_count,
            feature_shape=(
                len(feas),
                int(feature_lengths.max()) if feature_lengths.size else 0,
            ),
        )

    if label_column in res:
        physical_label_col = label_column
    else:
        try:
            physical_label_col = feature_layout.column_for(label_column)
        except ValueError:
            physical_label_col = label_column

    if physical_label_col not in res:
        raise ValueError(
            f"Training batch is missing the configured label column {label_column!r}"
        )
    out["labels"] = _system_i32_values(
        res[physical_label_col], column_name=label_column
    )

    for meta_col in metadata_cols:
        col_name = feature_layout.column_for(meta_col)
        if col_name not in res:
            raise ValueError(
                f"Metadata feature not found in training batch: logical={meta_col!r} "
                f"physical={col_name!r}; available={sorted(res)}"
            )
        col = res[col_name]
        out[meta_col] = _metadata_values(
            col,
            feature_layout.output_kind_for(meta_col),
            column_name=col_name,
            null_index_sentinel=feature_layout.null_index_sentinel,
        )

    return out


__all__ = [
    "NULL_FID_SENTINEL",
    "table_to_training_dict",
]
