from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingStateBlueprint:
    groups: tuple[str, ...]
    leaves: tuple[tuple[str, tuple[int, ...], str, str], ...]
    schema_digest: str

    def __post_init__(self) -> None:
        if not self.groups or len(self.groups) != len(set(self.groups)):
            raise ValueError(
                "TrainingStateBlueprint groups must be unique and non-empty"
            )
        if not self.leaves:
            raise ValueError("TrainingStateBlueprint requires at least one leaf")
        if not self.schema_digest:
            raise ValueError("TrainingStateBlueprint schema_digest must be non-empty")
        paths = tuple(leaf[0] for leaf in self.leaves)
        if len(paths) != len(set(paths)):
            raise ValueError("TrainingStateBlueprint leaf paths must be unique")
        for path, shape, dtype, role in self.leaves:
            if not path or any(int(dimension) < 0 for dimension in shape):
                raise ValueError("TrainingStateBlueprint leaf shape is invalid")
            if not dtype or not role:
                raise ValueError(
                    "TrainingStateBlueprint leaf dtype and role are required"
                )


class TrainingStateDefinition(ABC):
    @abstractmethod
    def abstract_state(self) -> Any:
        """Return structure and metadata without concrete trainable values."""

    @abstractmethod
    def initialize_values(self, *args: Any, **kwargs: Any) -> Any:
        """Create concrete values for a fresh training run."""


class TrainingStateLifecycle(str, Enum):
    """Explicit source of concrete values for a training state."""

    FRESH_INITIALIZE = "fresh_initialize"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    MODEL_WEIGHTS = "model_weights"


__all__ = [
    "TrainingStateBlueprint",
    "TrainingStateDefinition",
    "TrainingStateLifecycle",
]
