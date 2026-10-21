from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flare.data.pipeline.consumption.source.batching import BatchValidity


@dataclass(frozen=True)
class MetricBatch:
    data: Mapping[str, Any]
    validity: BatchValidity


@dataclass(frozen=True)
class MetricReference:
    ref_id: str
    handle: Any


@dataclass(frozen=True)
class MetricResult:
    values: Mapping[str, Any]
    metadata: Mapping[str, Any]


@runtime_checkable
class MetricSink(Protocol):
    def write(self, result: MetricResult) -> None: ...

    def close(self) -> None: ...


class MetricComputer(Protocol):
    def put_batch(self, batch: MetricBatch) -> MetricReference: ...
    def compute(self, refs: Sequence[MetricReference]) -> MetricResult: ...


@dataclass(frozen=True)
class MetricSpec:
    name: str
    target_names: tuple[str, ...] = ()
    higher_is_better: bool = True
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MetricSpec name cannot be empty")


__all__ = [
    "MetricBatch",
    "MetricComputer",
    "MetricReference",
    "MetricResult",
    "MetricSink",
    "MetricSpec",
]
