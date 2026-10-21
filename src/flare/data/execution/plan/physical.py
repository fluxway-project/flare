from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PhysicalStage:
    stage_id: str
    transform: type
    constructor_kwargs: dict[str, Any]
    min_actors: int
    max_actors: int
    tasks_in_flight: int
    num_cpus: float
    memory_bytes: int
    batch_size: int

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("physical stage id must not be empty")
        if self.min_actors <= 0 or self.max_actors < self.min_actors:
            raise ValueError(f"invalid actor bounds for stage {self.stage_id}")
        if self.tasks_in_flight <= 0 or self.num_cpus <= 0 or self.memory_bytes <= 0:
            raise ValueError(f"invalid resources for stage {self.stage_id}")
        if self.batch_size <= 0:
            raise ValueError(f"invalid batch size for stage {self.stage_id}")


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    stages: tuple[PhysicalStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("physical plan must contain stages")
        ids = tuple(stage.stage_id for stage in self.stages)
        if len(set(ids)) != len(ids):
            raise ValueError("physical stage ids must be unique")
