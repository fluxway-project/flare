from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from flare.training.contracts.ports import PortSpec
from flare.training.contracts.schema import BatchSchema


@dataclass(frozen=True)
class TaskSpec:
    name: str
    batch_schema: BatchSchema
    output_ports: tuple[PortSpec, ...]
    loss: Callable[[Any, Any], Any]
    metrics: Mapping[str, Any] | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"TaskSpec requires non-empty name, got {self.name!r}")
        if not isinstance(self.batch_schema, BatchSchema):
            raise TypeError(
                f"TaskSpec batch_schema must be BatchSchema, got {type(self.batch_schema).__name__}"
            )
        if not callable(self.loss):
            raise TypeError(
                f"TaskSpec loss must be callable, got {type(self.loss).__name__}"
            )


__all__ = ["TaskSpec"]
