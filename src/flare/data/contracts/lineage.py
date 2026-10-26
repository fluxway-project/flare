"""Asset-level facts and a provider-neutral batch export port."""

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from flare.data.contracts.assets import AssetRef
from flare.data.pipeline.production.publication.contract import validate_publication_id


@dataclass(frozen=True, slots=True)
class LineageContext:
    run_id: str
    operation_id: str

    def __post_init__(self) -> None:
        for value in (self.run_id, self.operation_id):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError("lineage run and operation ids must be canonical")


@dataclass(frozen=True, slots=True)
class InputRead:
    asset: AssetRef

    def __post_init__(self) -> None:
        if not isinstance(self.asset, AssetRef):
            raise TypeError("lineage input must identify an asset")


@dataclass(frozen=True, slots=True)
class OutputPublished:
    asset: AssetRef
    publication_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.asset, AssetRef):
            raise TypeError("lineage output must identify an asset")
        validate_publication_id(self.publication_id)


LineageEvent = InputRead | OutputPublished


@runtime_checkable
class LineageExporter(Protocol):
    def export(self, events: tuple[LineageEvent, ...]) -> None | Awaitable[None]: ...

    def close(self) -> None | Awaitable[None]: ...


@runtime_checkable
class LineageExporterFactory(Protocol):
    def create(
        self, context: LineageContext
    ) -> LineageExporter | Awaitable[LineageExporter]: ...


@dataclass(frozen=True, slots=True)
class LineageConfig:
    exporter_factory: LineageExporterFactory
    context: LineageContext
    max_pending_events: int = 1024
    coalesce_window: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.exporter_factory, LineageExporterFactory):
            raise TypeError("lineage configuration requires LineageExporterFactory")
        if not isinstance(self.context, LineageContext):
            raise TypeError("lineage configuration requires LineageContext")
        for value in (self.max_pending_events, self.coalesce_window):
            if type(value) is not int or value <= 0:
                raise ValueError("lineage metadata limits must be positive integers")
