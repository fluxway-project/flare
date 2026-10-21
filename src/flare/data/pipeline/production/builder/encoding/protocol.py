"""Encoder protocol — all encoders produce PyArrow Expr lists."""

from __future__ import annotations

from operator import index
from typing import Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.constants import NULL_FID_SENTINEL
from flare.data.schema.feature_catalog import FeatureOutputKind

SLOT_BITS_U64 = 48
MASK_48 = (1 << SLOT_BITS_U64) - 1
SPARSE_SLOT_COUNT = 1 << (64 - SLOT_BITS_U64)


def sparse_fid_i64(slot: int) -> int:
    slot_index = index(slot)
    if not 0 <= slot_index < SPARSE_SLOT_COUNT:
        raise ValueError(
            f"sparse slot must be in [0, {SPARSE_SLOT_COUNT}), got {slot_index}"
        )
    bits = slot_index << SLOT_BITS_U64
    return bits if bits < 1 << 63 else bits - (1 << 64)


def output_kind_pa_type(kind: FeatureOutputKind, length: int) -> pa.DataType:
    if kind is FeatureOutputKind.SPARSE_I64_FIDS:
        return pa.int64() if length == 1 else pa.list_(pa.int64())
    if kind is FeatureOutputKind.DENSE_I32_INDICES:
        return pa.int32() if length == 1 else pa.list_(pa.int32())
    if kind is FeatureOutputKind.DENSE_FLOAT:
        return pa.float32() if length == 1 else pa.list_(pa.float32())
    raise ValueError(f"unknown feature output kind: {kind}")


def encoded_default_for(kind: FeatureOutputKind, feature_code: int) -> int | float:
    if kind is FeatureOutputKind.SPARSE_I64_FIDS:
        return sparse_fid_i64(feature_code)
    if kind is FeatureOutputKind.DENSE_I32_INDICES:
        return 0
    if kind is FeatureOutputKind.DENSE_FLOAT:
        return 0.0
    raise ValueError(f"unknown feature output kind: {kind}")


def encoded_missing_for(kind: FeatureOutputKind, feature_code: int) -> int | float:
    if kind is FeatureOutputKind.SPARSE_I64_FIDS:
        return sparse_fid_i64(feature_code) + (NULL_FID_SENTINEL & MASK_48)
    if kind is FeatureOutputKind.DENSE_I32_INDICES:
        return NULL_FID_SENTINEL
    if kind is FeatureOutputKind.DENSE_FLOAT:
        return float(NULL_FID_SENTINEL)
    raise ValueError(f"unknown feature output kind: {kind}")


@runtime_checkable
class EncoderExpr(Protocol):
    """Feature encoder protocol — defines execution tiers for Acero AST or Table compute.

    列名约定:
      - SPARSE_I64_FIDS / DENSE_I32_INDICES -> ``fid_<feature_code>``
      - DENSE_FLOAT -> ``dense_<feature_code>``
    """

    @property
    def output_kind(self) -> FeatureOutputKind: ...

    def encoded_default(self, feature_code: int) -> int | float:
        return encoded_default_for(self.output_kind, feature_code)

    def encoded_missing(self, feature_code: int) -> int | float:
        return encoded_missing_for(self.output_kind, feature_code)

    @property
    def encoded_cardinality(self) -> int | None:
        return None

    def acero_expr(self, col_name: str, slot: int, length: int) -> pc.Expression | None:
        """Return a PyArrow compute expression for Tier 1 (Acero DAG) execution.

        Returns:
            A PyArrow Expression if this operation supports AST fusion, or None.
        """
        ...

    def table_compute(
        self, table: pa.Table, col_name: str, slot: int, length: int
    ) -> pa.ChunkedArray | None:
        """Execute Tier 2 (Vectorization) or Tier 3 (Numba) directly on the ChunkedArray.

        Args:
            table: The intermediate PyArrow Table produced after Tier 1.

        Returns:
            The computed pa.ChunkedArray, or None if handled by Tier 1.
        """
        return None
