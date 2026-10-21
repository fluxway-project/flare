from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import pyarrow as pa

from flare.data.contracts.schema import SchemaContract


@dataclass(frozen=True, slots=True)
class TransformContext:
    stage_id: str
    worker_id: str
    resources: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        invalid = {
            name: value
            for name, value in self.resources.items()
            if not name or value < 0
        }
        if invalid:
            raise ValueError(f"invalid transform resources: {invalid}")


@runtime_checkable
class BatchTransform(Protocol):
    @property
    def schema_contract(self) -> SchemaContract: ...

    def open(self, context: TransformContext) -> None: ...

    def apply(self, batch: pa.Table) -> pa.Table: ...

    def close(self) -> None: ...


TransformT_co = TypeVar("TransformT_co", bound=BatchTransform, covariant=True)


@runtime_checkable
class TransformFactory(Protocol[TransformT_co]):
    @property
    def transform_id(self) -> str: ...

    def create(self) -> TransformT_co: ...
