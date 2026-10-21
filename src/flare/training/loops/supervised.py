from __future__ import annotations

from typing import Any, Protocol


class SupervisedTrainingLoop(Protocol):
    """Protocol for synchronous/asynchronous supervised training loop."""

    def train_step(self, step_index: int, batch: Any) -> Any: ...
    def run(self, total_steps: int) -> None: ...


__all__ = ["SupervisedTrainingLoop"]
