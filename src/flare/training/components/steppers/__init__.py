from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flare.training.components.steppers.contracts import (
    ForwardFn,
    LossFn,
    StepperProtocol,
)


class GenericLossAdapter:
    def __init__(self, base_loss_fn: Callable[..., Any]) -> None:
        if not callable(base_loss_fn):
            raise TypeError("base_loss_fn must be callable")
        self.base_loss_fn = base_loss_fn

    def __call__(self, predictions: Any, targets: Mapping[str, Any]) -> Any:
        return self.base_loss_fn(predictions, targets)


__all__ = [
    "ForwardFn",
    "GenericLossAdapter",
    "LossFn",
    "StepperProtocol",
]
