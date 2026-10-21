from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flare.training.components.rl.contracts import Experience


@dataclass(frozen=True)
class ReplayBufferSpec:
    capacity: int
    batch_size: int
    prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4


@dataclass(frozen=True)
class RolloutBufferSpec:
    buffer_size: int
    gamma: float = 0.99
    gae_lambda: float = 0.95


class BufferProtocol(Protocol):
    def push(self, experience: Experience) -> None: ...
    def sample(self, batch_size: int) -> tuple[Any, ...]: ...
    def size(self) -> int: ...


__all__ = [
    "BufferProtocol",
    "ReplayBufferSpec",
    "RolloutBufferSpec",
]
