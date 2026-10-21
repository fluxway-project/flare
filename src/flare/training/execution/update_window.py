from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class OptimizerUpdateWindow:
    """Immutable optimizer-window identity shared by dense and sparse updates."""

    micro_step: int
    accumulation_steps: int
    at_boundary: bool
    should_apply_update: bool
    optimizer_step: int

    def __post_init__(self) -> None:
        if self.micro_step <= 0:
            raise ValueError("micro_step must be positive")
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        if self.micro_step > self.accumulation_steps:
            raise ValueError("micro_step cannot exceed accumulation_steps")
        if self.optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        expected_boundary = self.micro_step == self.accumulation_steps
        if self.at_boundary != expected_boundary:
            raise ValueError("at_boundary must equal the accumulation boundary state")
        if self.should_apply_update and not self.at_boundary:
            raise ValueError("an update can only apply at an accumulation boundary")

    def encode(self) -> tuple[int, int, int, int, int]:
        """Return the stable scalar payload used by Fluxway control edges."""
        return (
            self.micro_step,
            self.accumulation_steps,
            int(self.at_boundary),
            int(self.should_apply_update),
            self.optimizer_step,
        )

    @classmethod
    def decode(
        cls, values: Sequence[int | np.integer[Any]] | np.ndarray
    ) -> OptimizerUpdateWindow:
        if len(values) != 5:
            raise ValueError(
                "encoded OptimizerUpdateWindow must contain exactly five integers"
            )
        (
            micro_step,
            accumulation_steps,
            at_boundary,
            should_apply,
            optimizer_step,
        ) = (int(value) for value in values)
        if at_boundary not in (0, 1) or should_apply not in (0, 1):
            raise ValueError(
                "encoded OptimizerUpdateWindow boolean fields must be 0 or 1"
            )
        return cls(
            micro_step=micro_step,
            accumulation_steps=accumulation_steps,
            at_boundary=bool(at_boundary),
            should_apply_update=bool(should_apply),
            optimizer_step=optimizer_step,
        )


__all__ = ["OptimizerUpdateWindow"]
