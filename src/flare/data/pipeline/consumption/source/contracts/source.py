from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from flare.data.pipeline.consumption.source.contracts.model import (
    SourceRecordRange,
)
from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
)


@runtime_checkable
class ControlledSourceScanAdapter(Protocol):
    """Bounded source scan owner shared by controlled source formats."""

    @property
    def binding_identity(self) -> Mapping[str, str]: ...

    def initial_scan_cursor(self) -> OpaqueSourceCursor | None: ...

    def advance_cursor(
        self, cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor: ...

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult: ...


@runtime_checkable
class SourceReader(Protocol):
    def bind_source_identity(self, source_identity: str) -> None: ...

    def close(self) -> None | Awaitable[None]: ...

    def read_batches(
        self,
        record_range: SourceRecordRange,
        projection: Sequence[str],
        *,
        batch_size: int,
    ) -> AsyncIterator[Any] | object | Awaitable[object]: ...

    def read(
        self, record_range: SourceRecordRange, projection: Sequence[str]
    ) -> object | Awaitable[object]: ...


@runtime_checkable
class SourceFactory(Protocol):
    def create_scan_planner(
        self, options: Mapping[str, Any]
    ) -> ControlledSourceScanAdapter: ...

    def create_reader(self, options: Mapping[str, Any]) -> SourceReader: ...
