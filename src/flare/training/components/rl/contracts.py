from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    shape: tuple[int, ...]
    dtype: str = "int32"
    num_actions: int | None = None


@dataclass(frozen=True)
class RewardSpec:
    shape: tuple[int, ...] = (1,)
    dtype: str = "float32"


@dataclass(frozen=True)
class Experience:
    observations: Mapping[str, Any]
    actions: Any
    rewards: Any
    next_observations: Mapping[str, Any]
    dones: Any
    log_probs: Any | None = None
    values: Any | None = None


@dataclass(frozen=True)
class Trajectory:
    experiences: tuple[Experience, ...]
    total_reward: float
    length: int


__all__ = [
    "ActionSpec",
    "Experience",
    "RewardSpec",
    "Trajectory",
]
