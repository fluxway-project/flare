from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

ForwardFn = Callable[[Any, Mapping[str, Any]], Any]
LossFn = Callable[[Any, Mapping[str, Any]], Any]


class StepperProtocol(Protocol):
    """Generic protocol for forward-backward-step execution."""

    def step(
        self,
        module: Any,
        optimizer: Any,
        inputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        sample_weights: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]: ...


__all__ = ["ForwardFn", "LossFn", "StepperProtocol"]
