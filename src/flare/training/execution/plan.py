from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from flare.training.zero.contracts import ZeroStage


@dataclass(frozen=True, slots=True)
class ZeroExecutionPolicy:
    stage: ZeroStage
    axis_name: str = "dp"

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ZeroStage):
            raise TypeError("stage must be a ZeroStage instance")
        if not self.axis_name:
            raise ValueError("axis_name must be non-empty")


@dataclass(frozen=True, slots=True)
class GradientAccumulationPolicy:
    micro_steps: int = 1

    def __post_init__(self) -> None:
        if self.micro_steps <= 0:
            raise ValueError("micro_steps must be positive")


@dataclass(frozen=True, slots=True)
class MacroUpdatePolicy:
    apply_optimizer_at_boundary: bool = True


@dataclass(frozen=True, slots=True)
class TrainingExecutionPlan:
    """Explicit composition of partitioning, accumulation and update policy."""

    zero: ZeroExecutionPolicy
    accumulation: GradientAccumulationPolicy
    update: MacroUpdatePolicy

    @classmethod
    def replicated(
        cls,
        *,
        micro_steps: int = 1,
        apply_optimizer_at_boundary: bool = True,
    ) -> TrainingExecutionPlan:
        return cls(
            zero=ZeroExecutionPolicy(ZeroStage.REPLICATED),
            accumulation=GradientAccumulationPolicy(micro_steps),
            update=MacroUpdatePolicy(apply_optimizer_at_boundary),
        )


@dataclass(frozen=True, slots=True)
class TrainingInputPlan:
    mode: Literal["manifest", "repeat_first_batch"]
    manifest_source_enabled: bool
    source_epochs: int
    checkpoint_enabled: bool
    checkpoint_reason: Literal[
        "manifest_source", "existing_checkpoint", "scheduled", "disabled"
    ]


def resolve_training_input_plan(
    *,
    reader_repeat_first_batch: bool,
    latest_checkpoint_step: int | None,
    run_steps: int,
    target_step: int,
    checkpoint_interval_steps: int,
) -> TrainingInputPlan:
    if not isinstance(reader_repeat_first_batch, bool):
        raise TypeError(
            "reader_repeat_first_batch must be a boolean, "
            f"got {type(reader_repeat_first_batch).__name__}"
        )
    if run_steps <= 0:
        raise ValueError(f"run_steps must be positive, got {run_steps}")
    if target_step <= 0:
        raise ValueError(f"target_step must be positive, got {target_step}")
    if checkpoint_interval_steps <= 0:
        raise ValueError(
            "checkpoint_interval_steps must be positive, "
            f"got {checkpoint_interval_steps}"
        )
    if latest_checkpoint_step is not None and (
        isinstance(latest_checkpoint_step, bool) or latest_checkpoint_step < 0
    ):
        raise ValueError(
            "latest_checkpoint_step must be non-negative when provided, "
            f"got {latest_checkpoint_step}"
        )
    if reader_repeat_first_batch and latest_checkpoint_step is not None:
        raise ValueError(
            "reader_repeat_first_batch simulation mode cannot restore a checkpoint; "
            "remove --reader-repeat-first-batch for resumable training"
        )

    manifest_source_enabled = not reader_repeat_first_batch
    if latest_checkpoint_step is not None:
        checkpoint_enabled = True
        checkpoint_reason = "existing_checkpoint"
    elif checkpoint_interval_steps <= run_steps:
        checkpoint_enabled = True
        checkpoint_reason = "scheduled"
    else:
        checkpoint_enabled = False
        checkpoint_reason = "disabled"

    return TrainingInputPlan(
        mode="manifest" if manifest_source_enabled else "repeat_first_batch",
        manifest_source_enabled=manifest_source_enabled,
        source_epochs=1 if manifest_source_enabled else target_step,
        checkpoint_enabled=checkpoint_enabled,
        checkpoint_reason=checkpoint_reason,
    )


__all__ = [
    "GradientAccumulationPolicy",
    "MacroUpdatePolicy",
    "TrainingExecutionPlan",
    "TrainingInputPlan",
    "ZeroExecutionPolicy",
    "resolve_training_input_plan",
]
