from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.constants import (
    BUCKET_EPSILON,
    NULL_FID_SENTINEL,
)
from flare.data.pipeline.production.builder.encoding.base import (
    pack_sparse_i64_values,
    sparse_i64_acero_expr,
    sparse_i64_list_array,
)
from flare.data.pipeline.production.builder.encoding.protocol import (
    EncoderExpr,
)
from flare.data.schema.feature_catalog import FeatureOutputKind

_INT32_MAX = np.iinfo(np.int32).max


@numba.njit(nogil=True, cache=True)
def _numba_map_bucket(values, bounds, default_value, out):
    for i in range(len(values)):
        value = values[i]
        if value == NULL_FID_SENTINEL or value == default_value:
            out[i] = NULL_FID_SENTINEL
        else:
            out[i] = np.searchsorted(bounds, value, side="left")


@numba.njit(nogil=True, cache=True)
def _numba_map_fixed_width(values, width, default_value, out):
    for i in range(len(values)):
        value = values[i]
        if value == NULL_FID_SENTINEL or value == default_value:
            out[i] = NULL_FID_SENTINEL
        else:
            out[i] = int(value / width)


@numba.njit(nogil=True, cache=True)
def _numba_map_categorical(
    values, sorted_keys, sorted_vals, default_val, fallback_idx, out
):
    for i in range(len(values)):
        v = values[i]
        idx = np.searchsorted(sorted_keys, v)
        if idx < fallback_idx and sorted_keys[idx] == v:
            out[i] = sorted_vals[idx]
        else:
            out[i] = default_val


@dataclass(frozen=True, slots=True)
class BucketPartition:
    thresholds: tuple[float, ...]
    runtime_array: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, tuple):
            raise TypeError("bucket thresholds must be a tuple")
        if not isinstance(self.runtime_array, np.ndarray):
            raise TypeError("bucket runtime_array must be a NumPy array")
        if self.runtime_array.flags.writeable:
            raise ValueError("bucket runtime_array must be read-only")

    @classmethod
    def from_bounds(cls, bounds: Sequence[float]) -> BucketPartition:
        bounds_arr = np.sort(np.asarray(tuple(bounds), dtype=np.float32))
        if not np.all(np.isfinite(bounds_arr)):
            raise ValueError("bucket bounds must be finite")
        if len(bounds_arr) > 1:
            distinct = np.empty(len(bounds_arr), dtype=np.float32)
            distinct[0] = bounds_arr[0]
            count = 1
            for value in bounds_arr[1:]:
                if value - distinct[count - 1] >= BUCKET_EPSILON:
                    distinct[count] = value
                    count += 1
            bounds_arr = distinct[:count]
        negative = bounds_arr <= -10.0
        positive = bounds_arr >= 10.0
        middle = ~(negative | positive)
        bounds_arr[negative] *= np.float32(1.0 + BUCKET_EPSILON)
        bounds_arr[positive] *= np.float32(1.0 - BUCKET_EPSILON)
        bounds_arr[middle] -= np.float32(BUCKET_EPSILON)
        bounds_arr.flags.writeable = False
        return cls(
            thresholds=tuple(float(value) for value in bounds_arr),
            runtime_array=bounds_arr,
        )

    @property
    def cardinality(self) -> int:
        return len(self.thresholds) + 1


@dataclass(frozen=True, slots=True)
class BucketEncoderExpr(EncoderExpr):
    partition: BucketPartition
    in_dtype: str = "float32"
    default_value: int | float | None = None
    output_kind: FeatureOutputKind = FeatureOutputKind.DENSE_I32_INDICES

    def __post_init__(self) -> None:
        if not isinstance(self.partition, BucketPartition):
            raise TypeError("bucket partition must be a BucketPartition")

    @property
    def bounds(self) -> tuple[float, ...]:
        return self.partition.thresholds

    @property
    def encoded_cardinality(self) -> int:
        return self.partition.cardinality

    @property
    def _effective_default(self) -> int | float:
        is_int = self.in_dtype in ("int64", "int32")
        val = (
            self.default_value
            if self.default_value is not None
            else (NULL_FID_SENTINEL if is_int else 0.0)
        )
        return int(val) if is_int else float(val)

    def encoded_default(self, feature_code: int) -> int:
        return NULL_FID_SENTINEL

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        is_int = self.in_dtype in ("int64", "int32")
        pa_dtype = pa.int64() if is_int else pa.float32()
        is_list = length > 1

        arr = table.column(col_name).cast(
            pa.list_(pa_dtype) if is_list else pa_dtype, safe=False
        )

        def _process_chunk(col):
            flat = col.flatten() if is_list else col

            is_null = flat.is_null()
            if flat.null_count > 0:
                flat_filled = pc.fill_null(flat, pa.scalar(0, type=pa_dtype))
            else:
                flat_filled = flat

            flat_np = flat_filled.to_numpy(zero_copy_only=False)
            num_items = len(flat_np)
            out_buf = pa.allocate_buffer(num_items * 4)
            out_np = np.frombuffer(out_buf, dtype=np.int32)
            _numba_map_bucket(
                flat_np,
                self.partition.runtime_array,
                self._effective_default,
                out_np,
            )

            out_flat = pa.Array.from_buffers(pa.int32(), num_items, [None, out_buf])
            if flat.null_count > 0:
                out_flat = pc.call_function(
                    "if_else",
                    [is_null, pa.scalar(None, type=pa.int32()), out_flat],
                )

            if is_list:
                if isinstance(col.type, pa.FixedSizeListType):
                    return pa.FixedSizeListArray.from_arrays(out_flat, length)
                else:
                    shifted_offsets = pc.call_function(
                        "subtract", [col.offsets, col.offsets[0]]
                    )
                    return pa.ListArray.from_arrays(shifted_offsets, out_flat)
            else:
                return out_flat

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@dataclass(frozen=True, slots=True)
class FixedWidthBucketEncoderExpr(EncoderExpr):
    width: float
    in_dtype: str = "float32"
    default_value: int | float | None = None
    output_kind: FeatureOutputKind = FeatureOutputKind.DENSE_I32_INDICES

    def __post_init__(self) -> None:
        if not np.isfinite(self.width) or self.width <= 0:
            raise ValueError("fixed-width bucket width must be finite and positive")

    @property
    def _effective_default(self) -> int | float:
        is_int = self.in_dtype in ("int64", "int32")
        val = (
            self.default_value
            if self.default_value is not None
            else (NULL_FID_SENTINEL if is_int else 0.0)
        )
        return int(val) if is_int else float(val)

    def encoded_default(self, feature_code: int) -> int:
        return NULL_FID_SENTINEL

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        is_int = self.in_dtype in ("int64", "int32")
        pa_dtype = pa.int64() if is_int else pa.float32()
        is_list = length > 1
        arr = table.column(col_name).cast(
            pa.list_(pa_dtype) if is_list else pa_dtype, safe=False
        )

        def _process_chunk(col):
            flat = col.flatten() if is_list else col
            is_null = flat.is_null()
            flat_filled = (
                pc.fill_null(flat, pa.scalar(0, type=pa_dtype))
                if flat.null_count > 0
                else flat
            )
            values = flat_filled.to_numpy(zero_copy_only=False)
            output_buffer = pa.allocate_buffer(len(values) * 4)
            output = np.frombuffer(output_buffer, dtype=np.int32)
            _numba_map_fixed_width(
                values,
                self.width,
                self._effective_default,
                output,
            )
            result = pa.Array.from_buffers(
                pa.int32(), len(values), [None, output_buffer]
            )
            if flat.null_count > 0:
                result = pc.if_else(is_null, pa.scalar(None, type=pa.int32()), result)
            if not is_list:
                return result
            if isinstance(col.type, pa.FixedSizeListType):
                return pa.FixedSizeListArray.from_arrays(result, length)
            shifted_offsets = pc.subtract(col.offsets, col.offsets[0])
            return pa.ListArray.from_arrays(shifted_offsets, result)

        return pa.chunked_array([_process_chunk(chunk) for chunk in arr.chunks])


@dataclass(frozen=True, slots=True)
class CategoryLookupTable:
    mapping: MappingProxyType[int, int]
    sorted_keys: np.ndarray = field(repr=False, compare=False)
    sorted_vals: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, MappingProxyType):
            raise TypeError("category mapping must be a MappingProxyType")
        if not isinstance(self.sorted_keys, np.ndarray) or not isinstance(
            self.sorted_vals, np.ndarray
        ):
            raise TypeError("category derived arrays must be NumPy arrays")
        if self.sorted_keys.flags.writeable or self.sorted_vals.flags.writeable:
            raise ValueError("category derived arrays must be read-only")

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[int, int], default_index: int = 0
    ) -> CategoryLookupTable:
        canonical_mapping = MappingProxyType(
            {int(key): int(value) for key, value in mapping.items()}
        )
        indices = (default_index, *canonical_mapping.values())
        if min(indices) < 0 or max(indices) > _INT32_MAX:
            raise ValueError("category indices must fit the non-negative int32 domain")
        keys = np.array(list(canonical_mapping.keys()), dtype=np.int64)
        vals = np.array(list(canonical_mapping.values()), dtype=np.int32)
        sort_idx = np.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_vals = vals[sort_idx]
        sorted_keys.flags.writeable = False
        sorted_vals.flags.writeable = False
        return cls(
            mapping=canonical_mapping,
            sorted_keys=sorted_keys,
            sorted_vals=sorted_vals,
        )


@dataclass(frozen=True, slots=True)
class CategoryEncoderExpr(EncoderExpr):
    table: CategoryLookupTable
    default_index: int = 0
    output_kind: FeatureOutputKind = FeatureOutputKind.DENSE_I32_INDICES

    def __post_init__(self) -> None:
        if not isinstance(self.table, CategoryLookupTable):
            raise TypeError("table must be a CategoryLookupTable")

    @property
    def mapping(self) -> MappingProxyType[int, int]:
        return self.table.mapping

    def encoded_default(self, feature_code: int) -> int:
        return self.default_index

    @property
    def encoded_cardinality(self) -> int:
        return max((self.default_index, *self.table.mapping.values())) + 1

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(
        self,
        table: pa.Table,
        col_name: str,
        slot: int,
        length: int,
    ) -> pa.ChunkedArray | None:
        is_list = length > 1
        arr = table.column(col_name).cast(
            pa.list_(pa.int64()) if is_list else pa.int64(), safe=False
        )
        fallback_idx = len(self.table.sorted_keys)

        def _process_chunk(col):
            flat = col.flatten() if is_list else col

            if flat.null_count > 0:
                flat_filled = pc.fill_null(flat, 0)
            else:
                flat_filled = flat

            flat_np = flat_filled.to_numpy(zero_copy_only=False)
            num_items = len(flat_np)
            out_buf = pa.allocate_buffer(num_items * 4)
            out_np = np.frombuffer(out_buf, dtype=np.int32)
            _numba_map_categorical(
                flat_np,
                self.table.sorted_keys,
                self.table.sorted_vals,
                self.default_index,
                fallback_idx,
                out_np,
            )

            out_flat = pa.Array.from_buffers(pa.int32(), num_items, [None, out_buf])

            if is_list:
                if isinstance(col.type, pa.FixedSizeListType):
                    return pa.FixedSizeListArray.from_arrays(out_flat, length)
                else:
                    shifted_offsets = pc.call_function(
                        "subtract", [col.offsets, col.offsets[0]]
                    )
                    return pa.ListArray.from_arrays(shifted_offsets, out_flat)
            else:
                return out_flat

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@dataclass(frozen=True, slots=True)
class AutoCategoryEncoderExpr(EncoderExpr):
    output_kind: FeatureOutputKind = FeatureOutputKind.SPARSE_I64_FIDS

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return sparse_i64_acero_expr(col_name, slot, length)

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        is_list = length > 1
        arr = table.column(col_name).cast(
            pa.list_(pa.int64()) if is_list else pa.int64(), safe=False
        )

        def _process_chunk(col):
            flat = col.flatten() if is_list else col

            packed_arrow = pack_sparse_i64_values(flat, slot)

            if is_list:
                return sparse_i64_list_array(col, packed_arrow, slot, length)
            else:
                return packed_arrow

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@dataclass(frozen=True, slots=True)
class RawFloatEncoderExpr(EncoderExpr):
    output_kind: FeatureOutputKind = FeatureOutputKind.DENSE_FLOAT

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        if length == 1:
            return pc.field(col_name).cast(pa.float32(), safe=False)
        return pc.field(col_name)

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        if length == 1:
            return table.column(col_name).cast(pa.float32(), safe=False)
        return table.column(col_name).cast(pa.list_(pa.float32()), safe=False)
