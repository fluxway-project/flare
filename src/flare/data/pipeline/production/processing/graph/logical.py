from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class LogicalTransform:
    node_id: str
    factory_id: str
    dependencies: tuple[str, ...]
    parameters: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("logical transform node_id must not be empty")
        if not self.factory_id:
            raise ValueError("logical transform factory_id must not be empty")
        if self.node_id in self.dependencies:
            raise ValueError(
                f"logical transform cannot depend on itself: {self.node_id}"
            )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(
                f"logical transform dependencies must be unique: {self.node_id}"
            )


@dataclass(frozen=True, slots=True)
class LogicalPipeline:
    pipeline_id: str
    transforms: tuple[LogicalTransform, ...]

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("logical pipeline id must not be empty")
        if not self.transforms:
            raise ValueError("logical pipeline must contain transforms")
        node_ids = tuple(transform.node_id for transform in self.transforms)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("logical pipeline node ids must be unique")
