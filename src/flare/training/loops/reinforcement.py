from __future__ import annotations

from typing import Any, Protocol


class ReinforcementTrainingLoop(Protocol):
    """Protocol for asynchronous/synchronous reinforcement learning training loop."""

    def rollout_step(self, num_episodes: int) -> Any: ...
    def update_policy(self, batch: Any) -> Any: ...
    def sync_weights(self) -> None: ...
    def run(self, total_iterations: int) -> None: ...


__all__ = ["ReinforcementTrainingLoop"]
