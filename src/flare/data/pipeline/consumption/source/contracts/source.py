"""Source execution contracts; task payloads belong to the source backend."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceWorkItem,
)

from flare.data.contracts.assets import AssetRef


@dataclass(frozen=True, slots=True)
class SourceBatch:
    payload: object
    cardinality: int
    inputs: tuple[AssetRef, ...] = ()

    def __post_init__(self) -> None:
        if self.payload is None:
            raise ValueError("source batch payload must not be None")
        if type(self.cardinality) is not int or self.cardinality < 0:
            raise ValueError("source batch cardinality must be non-negative")
        if not isinstance(self.inputs, tuple) or any(
            not isinstance(asset, AssetRef) for asset in self.inputs
        ):
            raise TypeError("source batch inputs must be an AssetRef tuple")


@runtime_checkable
class ControlledSourceScanAdapter(Protocol):
    @property
    def binding_identity(self) -> Mapping[str, str]: ...

    def initial_scan_cursor(
        self,
    ) -> OpaqueSourceCursor | None | Awaitable[OpaqueSourceCursor | None]: ...

    def advance_cursor(
        self, cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor: ...

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult | Awaitable[SourceScanResult]: ...

    def close(self) -> None | Awaitable[None]: ...


@runtime_checkable
class SourceReadSession(Protocol):
    def read(
        self, step: int, *, work_items: Sequence[SourceWorkItem]
    ) -> SourceBatch | Awaitable[SourceBatch]: ...

    def close(self) -> None | Awaitable[None]: ...


@runtime_checkable
class SourceFactory(Protocol):
    def prepare_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def create_scan_planner(
        self, options: Mapping[str, Any]
    ) -> ControlledSourceScanAdapter | Awaitable[ControlledSourceScanAdapter]: ...

    def create_reader(
        self, options: Mapping[str, Any], *, include_lineage: bool = False
    ) -> SourceReadSession | Awaitable[SourceReadSession]: ...
