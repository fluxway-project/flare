from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricWindow:
    prediction_refs: tuple[Any, ...]
    loss_sum: float
    sampled_step_count: int
    step_count: int
    trained_sample_count: int
    global_step: int
    elapsed_sec: float

    @property
    def throughput_steps(self) -> float:
        return self.step_count / self.elapsed_sec

    @property
    def throughput_samples(self) -> float:
        return self.trained_sample_count / self.elapsed_sec

    def __post_init__(self) -> None:
        if self.step_count <= 0:
            raise ValueError("MetricWindow step_count must be positive")
        if self.trained_sample_count <= 0:
            raise ValueError("MetricWindow trained_sample_count must be positive")
        if self.elapsed_sec <= 0:
            raise ValueError("MetricWindow elapsed_sec must be positive")
        if self.prediction_refs and self.sampled_step_count <= 0:
            raise ValueError(
                "MetricWindow with prediction refs must have sampled steps"
            )
        if not self.prediction_refs and self.sampled_step_count != 0:
            raise ValueError("MetricWindow sampled steps must have prediction refs")


__all__ = ["MetricWindow"]
