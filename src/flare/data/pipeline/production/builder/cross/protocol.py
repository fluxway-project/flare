"""Cross-feature protocol — FeaCross (pre-encode) and FidCross (post-encode)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class CrossColumnContext(Protocol):
    """Read-only columns available to a cross operator."""

    def column(self, name: str) -> pa.ChunkedArray: ...


class CrossComputeContext(CrossColumnContext, Protocol):
    """Column access and cache services for vectorized FeaCross operators."""

    def cast(self, name: str, typ: pa.DataType) -> pa.ChunkedArray: ...

    def flat_numpy(
        self,
        name: str,
        typ: pa.DataType,
        chunk_idx: int,
        np_type: type[np.generic],
        *,
        list_mode: bool,
    ) -> np.ndarray: ...

    def offsets(self, name: str, typ: pa.DataType, chunk_idx: int) -> np.ndarray: ...


@runtime_checkable
class FeaCrossExpr(Protocol):
    """Pre-encode cross: produces new raw feature columns before encoding.

    产生的列之后会被 EncoderExpr 编码为 fid/float。
    输出列名 = self.name。
    """

    @property
    def name(self) -> str: ...

    @property
    def dep_names(self) -> list[str]: ...

    def acero_expr(self) -> pc.Expression | None:
        """Return expression producing raw feature column for Tier 1 Acero DAG."""
        ...

    def table_compute(self, context: CrossComputeContext) -> pa.ChunkedArray | None:
        """Execute Tier 2 (Vectorization) or Tier 3 (Numba) directly on the Table."""
        return None


@runtime_checkable
class FidCrossExpr(Protocol):
    """Post-encode cross: produces new fid columns from already-encoded fid columns.

    输入和输出都是已编码的 int64 FID bit pattern。
    输出列名 = self.name。
    """

    @property
    def name(self) -> str: ...

    @property
    def dep_names(self) -> list[str]: ...

    @property
    def slot(self) -> int: ...

    def acero_expr(self) -> pc.Expression | None:
        """Return expression producing an int64 FID column for Tier 1 Acero DAG."""
        ...

    def table_compute(self, context: CrossColumnContext) -> pa.ChunkedArray | None:
        """Execute Tier 2 (Vectorization) or Tier 3 (Numba) directly on the Table."""
        return None
