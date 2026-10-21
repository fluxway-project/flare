"""Encoding base — re-exports and shared constants."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.encoding.protocol import (
    MASK_48,
    SLOT_BITS_U64,
    EncoderExpr,
    sparse_fid_i64,
)
from flare.data.schema.feature_catalog import FeatureOutputKind


def sparse_bits_u64_to_i64(
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    if values.type != pa.uint64():
        raise TypeError(f"sparse arithmetic bits must be uint64, got {values.type}")
    if isinstance(values, pa.ChunkedArray):
        return pa.chunked_array([chunk.view(pa.int64()) for chunk in values.chunks])
    return values.view(pa.int64())


def pack_sparse_i64_low_bits(flat_i64, slot: int):
    if flat_i64.type != pa.int64():
        raise TypeError(f"sparse low bits must be int64, got {flat_i64.type}")
    masked = pc.bit_wise_and(  # pyrefly: ignore[missing-attribute]
        flat_i64, pa.scalar(MASK_48, type=pa.int64())
    )
    return pc.add(  # pyrefly: ignore[missing-attribute]
        masked,
        pa.scalar(sparse_fid_i64(slot), type=pa.int64()),
    )


def pack_sparse_i64_values(flat, slot: int):
    if not pa.types.is_signed_integer(flat.type):
        raise TypeError(f"raw sparse values must use a signed integer, got {flat.type}")
    return pack_sparse_i64_low_bits(pc.cast(flat, pa.int64(), safe=False), slot)


def pack_sparse_u64_hashes(flat, slot: int):
    return pack_sparse_i64_low_bits(sparse_bits_u64_to_i64(flat), slot)


def sparse_i64_acero_expr(col_name: str, slot: int, length: int):
    if length != 1:
        return None
    values = pc.field(col_name).cast(pa.int64(), safe=False)
    masked = pc.bit_wise_and(  # pyrefly: ignore[missing-attribute]
        values, pa.scalar(MASK_48, type=pa.int64())
    )
    return pc.add(  # pyrefly: ignore[missing-attribute]
        masked,
        pa.scalar(sparse_fid_i64(slot), type=pa.int64()),
    )


def sparse_i64_list_array(col, packed_arrow, slot: int, length: int):
    if pa.types.is_fixed_size_list(col.type):
        return pa.FixedSizeListArray.from_arrays(packed_arrow, length)
    shifted_offsets = pc.subtract(  # pyrefly: ignore[missing-attribute]
        col.offsets, col.offsets[0]
    )
    return sparse_i64_list_array_from_offsets(
        shifted_offsets,
        packed_arrow,
        slot,
    )


def sparse_i64_list_array_from_offsets(offsets_arrow, packed_arrow, slot: int):
    return pa.ListArray.from_arrays(offsets_arrow, packed_arrow)


class SchemaOnlyEncoderExpr(EncoderExpr):
    """An encoder that produces no execution expressions, used solely to register
    the output schema for features computed elsewhere (e.g., fid cross features).
    """

    requires_base_encode: bool = False

    def __init__(self, output_kind: FeatureOutputKind):
        self._output_kind = output_kind

    @property
    def output_kind(self) -> FeatureOutputKind:
        return self._output_kind

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        return None

    def table_compute(self, table, col_name: str, slot: int, length: int):
        return None


__all__ = [
    "MASK_48",
    "SLOT_BITS_U64",
    "EncoderExpr",
    "SchemaOnlyEncoderExpr",
    "pack_sparse_i64_low_bits",
    "pack_sparse_i64_values",
    "pack_sparse_u64_hashes",
    "sparse_bits_u64_to_i64",
    "sparse_i64_acero_expr",
    "sparse_i64_list_array",
    "sparse_i64_list_array_from_offsets",
]
