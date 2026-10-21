from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from flax import nnx

from flare.runtime.diagnostics.state_audit import fingerprint_state
from flare.training.execution.plan import TrainingExecutionPlan
from flare.training.execution.update_window import OptimizerUpdateWindow
from flare.training.zero.contracts import ZeroStage

if TYPE_CHECKING:
    from flare.jax.training.engines.contracts import TrainStepResult
    from flare.jax.training.engines.replicated import ReplicatedNnxTrainEngine


class TrainStepExecutor(Protocol):
    """Execution seam consumed by domain-specific SPMD workers."""

    plan: TrainingExecutionPlan

    def prepare_execution(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> None: ...

    def micro_step(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[TrainStepResult, OptimizerUpdateWindow]: ...

    def apply_update(self, window: OptimizerUpdateWindow) -> bool: ...

    def finalize_checkpoint_window(self) -> bool: ...

    def current_micro_step(self) -> int: ...

    def capture_checkpoint_state(self) -> Mapping[str, Any]: ...

    def checkpoint_layout(self) -> Mapping[str, Any]: ...

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None: ...

    def capture_module_fingerprint(self) -> Mapping[str, Any]: ...

    def capture_audit_state(self) -> Mapping[str, Any]: ...


class ReplicatedTrainStepExecutor:
    """Concrete replicated executor with explicit accumulation boundaries."""

    def __init__(
        self, engine: ReplicatedNnxTrainEngine, plan: TrainingExecutionPlan
    ) -> None:
        if plan.zero.stage is not ZeroStage.REPLICATED:
            raise ValueError(
                "ReplicatedTrainStepExecutor requires ZeRO replicated policy"
            )
        if engine.grad_accum_steps != plan.accumulation.micro_steps:
            raise ValueError(
                "engine.grad_accum_steps must equal the execution plan accumulation policy"
            )
        self.engine = engine
        self.plan = plan

    def prepare_execution(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> None:
        del inputs, targets

    def micro_step(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[TrainStepResult, OptimizerUpdateWindow]:
        result = self.engine.train_micro_step(inputs, targets)
        window = OptimizerUpdateWindow(
            micro_step=result.micro_step,
            accumulation_steps=self.plan.accumulation.micro_steps,
            at_boundary=result.optimizer_applied,
            should_apply_update=result.optimizer_applied
            and self.plan.update.apply_optimizer_at_boundary,
            optimizer_step=self.engine.step,
        )
        return result, window

    def apply_update(self, window: OptimizerUpdateWindow) -> bool:
        self._validate_window(window)
        if not window.should_apply_update:
            return False
        self.engine.apply_accumulated_update()
        return True

    def current_micro_step(self) -> int:
        return int(self.engine.micro_step)

    def finalize_checkpoint_window(self) -> bool:
        return self.engine.finalize_checkpoint_window()

    def capture_checkpoint_state(self) -> Mapping[str, Any]:
        return self.engine.capture_checkpoint_state()

    def checkpoint_layout(self) -> Mapping[str, Any]:
        return self.engine.checkpoint_layout()

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.engine.restore_checkpoint_state(state)

    def capture_module_fingerprint(self) -> Mapping[str, Any]:
        if self.engine.module is None:
            raise RuntimeError("training module is not initialized")
        return fingerprint_state(nnx.state(self.engine.module))

    def capture_audit_state(self) -> Mapping[str, Any]:
        if self.engine.module is None or self.engine.optimizer is None:
            raise RuntimeError("training state is not initialized")
        return {
            "module": fingerprint_state(nnx.state(self.engine.module)),
            "optimizer": fingerprint_state(nnx.state(self.engine.optimizer)),
            "accumulated_gradients": (
                None
                if self.engine.accumulated_gradients is None
                else fingerprint_state(self.engine.accumulated_gradients)
            ),
            "raw_gradients": (
                None
                if self.engine.last_gradients is None
                else fingerprint_state(self.engine.last_gradients)
            ),
            "reduced_gradients": (
                None
                if self.engine.last_gradients is None
                else fingerprint_state(self.engine.last_gradients)
            ),
            "micro_step": self.engine.micro_step,
            "grad_accum_steps": self.engine.grad_accum_steps,
        }

    def _validate_window(self, window: OptimizerUpdateWindow) -> None:
        if window.optimizer_step != self.engine.step:
            raise ValueError(
                "optimizer step mismatch: "
                f"actual={window.optimizer_step}, expected={self.engine.step}"
            )


class DistributedPrepareExecution(Protocol):
    def __call__(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> None: ...


class DistributedMicroStep(Protocol):
    def __call__(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> TrainStepResult: ...


class DistributedMacroUpdate(Protocol):
    def __call__(self, window: OptimizerUpdateWindow) -> bool: ...


class DistributedCheckpointFinalize(Protocol):
    def __call__(self) -> bool: ...


class DistributedTrainStepExecutor:
    """Executor seam for an explicitly constructed ZeRO stepper.

    The distributed JAX function is injected by the caller. There is no
    replicated fallback and no stage-dependent branch in the micro-step path.
    """

    executor_kind = "distributed"

    def __init__(
        self,
        *,
        plan: TrainingExecutionPlan,
        stage: ZeroStage,
        prepare_execution: DistributedPrepareExecution | None = None,
        micro_step: DistributedMicroStep,
        micro_step_counter: Callable[[], int],
        macro_update: DistributedMacroUpdate,
        checkpoint_finalize: DistributedCheckpointFinalize,
        checkpoint_capture: Callable[[], Mapping[str, Any]],
        checkpoint_layout: Callable[[], Mapping[str, Any]],
        checkpoint_restore: Callable[[Mapping[str, Any]], None],
        audit_state_capture: Callable[[], Mapping[str, Any]],
        module_fingerprint_capture: Callable[[], Mapping[str, Any]],
    ) -> None:
        if plan.zero.stage is not stage:
            raise ValueError(
                f"distributed executor stage mismatch: expected={stage.name}, "
                f"got={plan.zero.stage.name}"
            )
        self.plan = plan
        self.stage = stage
        self.executor_kind = f"zero{stage.value}"
        self._prepare_execution = prepare_execution
        self._micro_step = micro_step
        self._micro_step_counter = micro_step_counter
        self._macro_update = macro_update
        self._checkpoint_finalize = checkpoint_finalize
        self._checkpoint_capture = checkpoint_capture
        self._checkpoint_layout = checkpoint_layout
        self._checkpoint_restore = checkpoint_restore
        self._audit_state_capture = audit_state_capture
        self._module_fingerprint_capture = module_fingerprint_capture
        self._optimizer_step: int | None = None

    def prepare_execution(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> None:
        if self._prepare_execution is not None:
            self._prepare_execution(inputs, targets)

    def micro_step(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[TrainStepResult, OptimizerUpdateWindow]:
        result = self._micro_step(inputs, targets)
        self._optimizer_step = result.step
        window = OptimizerUpdateWindow(
            micro_step=result.micro_step,
            accumulation_steps=self.plan.accumulation.micro_steps,
            at_boundary=result.optimizer_applied,
            should_apply_update=result.optimizer_applied
            and self.plan.update.apply_optimizer_at_boundary,
            optimizer_step=result.step,
        )
        return result, window

    def apply_update(self, window: OptimizerUpdateWindow) -> bool:
        if self._optimizer_step is None:
            raise RuntimeError(
                "optimizer update window requires a preceding micro-step"
            )
        if window.optimizer_step != self._optimizer_step:
            raise ValueError(
                "optimizer step mismatch: "
                f"actual={window.optimizer_step}, expected={self._optimizer_step}"
            )
        if not window.should_apply_update:
            return False
        applied = self._macro_update(window)
        if not isinstance(applied, bool):
            raise TypeError("distributed macro_update must return bool")
        return applied

    def current_micro_step(self) -> int:
        value = self._micro_step_counter()
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "distributed runtime must expose an integer micro_step counter"
            )
        return value

    def finalize_checkpoint_window(self) -> bool:
        finalized = self._checkpoint_finalize()
        if not isinstance(finalized, bool):
            raise TypeError("distributed checkpoint finalizer must return bool")
        return finalized

    def capture_checkpoint_state(self) -> Mapping[str, Any]:
        return dict(self._checkpoint_capture())

    def checkpoint_layout(self) -> Mapping[str, Any]:
        return dict(self._checkpoint_layout())

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self._checkpoint_restore(state)
        self._optimizer_step = int(state["step"])
        if self._optimizer_step < 0:
            raise ValueError("checkpoint step must be non-negative")

    def capture_module_fingerprint(self) -> Mapping[str, Any]:
        return dict(self._module_fingerprint_capture())

    def capture_audit_state(self) -> Mapping[str, Any]:
        return dict(self._audit_state_capture())


__all__ = [
    "DistributedTrainStepExecutor",
    "ReplicatedTrainStepExecutor",
    "TrainStepExecutor",
]
