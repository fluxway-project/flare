from dataclasses import dataclass

import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.constants import DEFAULT_NULL_SENTINEL
from flare.data.pipeline.production.builder.encoding.base import (
    pack_sparse_i64_values,
    pack_sparse_u64_hashes,
    sparse_i64_acero_expr,
    sparse_i64_list_array,
)
from flare.data.pipeline.production.builder.encoding.protocol import (
    EncoderExpr,
    encoded_missing_for,
    sparse_fid_i64,
)
from flare.data.schema.feature_catalog import FeatureOutputKind


@dataclass(frozen=True, slots=True)
class AsItIsEncoderExpr(EncoderExpr):
    output_kind: FeatureOutputKind = FeatureOutputKind.SPARSE_I64_FIDS

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return sparse_i64_acero_expr(col_name, slot, length)

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        is_list = length > 1
        arr = table.column(col_name)

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
class StringHashEncoderExpr(EncoderExpr):
    output_kind: FeatureOutputKind = FeatureOutputKind.SPARSE_I64_FIDS

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        arr = table.column(col_name).cast(
            pa.list_(pa.int64()) if length == 1 else pa.list_(pa.list_(pa.int64())),
            safe=False,
        )

        def _process_chunk(col):
            is_list = length > 1
            flat = col.flatten() if is_list else col

            import numpy as np

            flat_values = flat.flatten().to_numpy(zero_copy_only=False).astype(np.int64)
            offsets = np.asarray(flat.offsets).astype(np.int32)
            num_items = len(flat)
            out_buf = pa.allocate_buffer(num_items * 8)
            out_hashes = np.frombuffer(out_buf, dtype=np.uint64)
            _numba_fnv1a_list(flat_values, offsets, out_hashes, 0)

            if flat.null_count > 0:
                is_null = flat.is_null()
                null_mask = is_null.to_numpy(zero_copy_only=False)
                out_hashes[null_mask] = 0

            flat_uint64 = pa.Array.from_buffers(pa.uint64(), num_items, [None, out_buf])
            packed_arrow = pack_sparse_u64_hashes(flat_uint64, slot)

            if is_list:
                return sparse_i64_list_array(col, packed_arrow, slot, length)
            return packed_arrow

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@dataclass(frozen=True, slots=True)
class IdEncoderExpr(EncoderExpr):
    in_dtype: str
    hash_seed: int
    output_kind: FeatureOutputKind = FeatureOutputKind.SPARSE_I64_FIDS

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        import numpy as np

        is_int = self.in_dtype in ("int64", "int32")
        in_cast = pa.int64() if is_int else pa.float32()
        arr = table.column(col_name).cast(
            in_cast if length == 1 else pa.list_(in_cast), safe=False
        )

        def _process_chunk(col):
            is_list = length > 1
            flat = col.flatten() if is_list else col

            if flat.null_count > 0:
                flat_filled = pc.fill_null(flat, 0)
            else:
                flat_filled = flat

            flat_values = flat_filled.to_numpy(zero_copy_only=False).astype(np.int64)
            num_items = len(flat)
            out_buf = pa.allocate_buffer(num_items * 8)
            out_hashes = np.frombuffer(out_buf, dtype=np.uint64)
            _numba_fnv1a_scalar_seed(flat_values, self.hash_seed, out_hashes)

            # null handling
            if flat.null_count > 0:
                is_null = flat.is_null()
                null_mask = is_null.to_numpy(zero_copy_only=False)
                out_hashes[null_mask] = 0

            flat_uint64 = pa.Array.from_buffers(pa.uint64(), num_items, [None, out_buf])
            packed_arrow = pack_sparse_u64_hashes(flat_uint64, slot)

            if is_list:
                return sparse_i64_list_array(col, packed_arrow, slot, length)
            return packed_arrow

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@dataclass(frozen=True, slots=True)
class HashEncoderExpr(EncoderExpr):
    bucket_size: int
    default_value: int = DEFAULT_NULL_SENTINEL
    output_kind: FeatureOutputKind = FeatureOutputKind.SPARSE_I64_FIDS

    def __post_init__(self) -> None:
        if self.bucket_size <= 0:
            raise ValueError("hash bucket size must be positive")

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        arr = table.column(col_name).cast(
            pa.int64() if length == 1 else pa.list_(pa.int64()),
            safe=False,
        )

        def _process_chunk(col):
            is_list = length > 1
            flat = col.flatten() if is_list else col

            flat_values = flat.to_numpy(zero_copy_only=False).astype(np.int64)
            num_items = len(flat)
            out_buf = pa.allocate_buffer(num_items * 8)
            out_values = np.frombuffer(out_buf, dtype=np.int64)
            _numba_hash_bucket(
                flat_values,
                self.bucket_size,
                self.default_value,
                out_values,
            )

            if flat.null_count > 0:
                is_null = flat.is_null()
                null_mask = is_null.to_numpy(zero_copy_only=False)
                out_values[null_mask] = self.default_value

            flat_int64 = pa.Array.from_buffers(pa.int64(), num_items, [None, out_buf])
            packed_arrow = pack_sparse_i64_values(flat_int64, slot)

            if is_list:
                return sparse_i64_list_array(col, packed_arrow, slot, length)
            return packed_arrow

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@numba.njit(nogil=True, cache=True)
def _numba_hash_bucket(values, bucket_size, default_value, out):
    for i in range(len(values)):
        value = values[i]
        if value == default_value or value == DEFAULT_NULL_SENTINEL:
            out[i] = DEFAULT_NULL_SENTINEL
        else:
            out[i] = np.int64(np.uint64(value) % np.uint64(bucket_size))


@dataclass(frozen=True, slots=True)
class SharedEmbEncoderExpr(EncoderExpr):
    target_slot: int
    output_kind: FeatureOutputKind = FeatureOutputKind.SPARSE_I64_FIDS

    def encoded_default(self, feature_code: int) -> int:
        return sparse_fid_i64(self.target_slot)

    def encoded_missing(self, feature_code: int) -> int:
        return int(encoded_missing_for(self.output_kind, self.target_slot))

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return sparse_i64_acero_expr(col_name, self.target_slot, length)

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        is_list = length > 1
        arr = table.column(col_name)

        def _process_chunk(col):
            flat = col.flatten() if is_list else col
            packed_arrow = pack_sparse_i64_values(flat, self.target_slot)

            if is_list:
                return sparse_i64_list_array(
                    col, packed_arrow, self.target_slot, length
                )
            else:
                return packed_arrow

        res_chunks = []
        for chunk in arr.chunks:
            res_chunks.append(_process_chunk(chunk))
        return pa.chunked_array(res_chunks)


@numba.njit(nogil=True, cache=True)
def _numba_fnv1a_list(values, offsets, out, bucket_size=0):
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        h = np.uint64(14695981039346656037)
        for j in range(start, end):
            v = values[j]
            h ^= np.uint64(v)
            h *= np.uint64(1099511628211)
        if bucket_size > 0:
            out[i] = h % np.uint64(bucket_size)
        else:
            out[i] = h


@numba.njit(nogil=True, cache=True)
def _numba_fnv1a_scalar_seed(values, seed, out):
    for i in range(len(values)):
        v = values[i]
        h = np.uint64(14695981039346656037)
        s = seed
        while s > 0:
            h ^= np.uint64(s & 0xFF)
            h *= np.uint64(1099511628211)
            s >>= 8
        s = v
        if s == 0:
            h ^= np.uint64(0)
            h *= np.uint64(1099511628211)
        else:
            while s > 0:
                h ^= np.uint64(s & 0xFF)
                h *= np.uint64(1099511628211)
                s >>= 8
        out[i] = h
