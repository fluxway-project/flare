from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainStepResult:
    loss: Any
    predictions: Any
    gradients: Any
    input_gradients: Any
    step: int
    micro_step: int
    optimizer_applied: bool


__all__ = ["TrainStepResult"]
