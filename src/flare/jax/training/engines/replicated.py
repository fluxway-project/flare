from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.mesh import ParallelMesh, state_with_sharding
from flare.jax.training.engines.contracts import TrainStepResult
from flare.jax.training.gradients.accumulation import (
    accumulate_gradients,
    build_donated_gradient_accumulator,
)
from flare.jax.training.gradients.input import (
    partition_input_gradients,
    validate_input_gradient_keys,
)
from flare.jax.training.gradients.normalization import (
    mean_accumulated_gradients,
)
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.steppers import (
    compute_loss_and_gradients,
    compute_module_and_input_gradients,
)
from flare.training.state import TrainingStateLifecycle
from flare.training.zero.contracts import (
    ZeroPartitionContract,
    ZeroStage,
    zero_partition_contract,
)


class ReplicatedNnxTrainEngine:
    """JAX module training with explicit ZeRO semantics."""

    def __init__(
        self,
        state_definition: NnxTrainingStateDefinition,
        forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
        loss_fn: Callable[[Any, Mapping[str, Any]], Any],
        *,
        zero_stage: ZeroStage = ZeroStage.REPLICATED,
        grad_accum_steps: int = 1,
        mesh: Mesh | ParallelMesh | None = None,
        state_lifecycle: TrainingStateLifecycle = TrainingStateLifecycle.FRESH_INITIALIZE,
        materialize_parameter_gradients: bool = True,
        input_gradient_keys: tuple[str, ...] | None = None,
    ) -> None:
        if not callable(forward_fn):
            raise TypeError("forward_fn must be callable")
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        if grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {grad_accum_steps}"
            )
        if not isinstance(state_definition, NnxTrainingStateDefinition):
            raise TypeError(
                "state_definition must be NnxTrainingStateDefinition, "
                f"got {type(state_definition).__name__}"
            )
        self.state_definition = state_definition
        if not isinstance(state_lifecycle, TrainingStateLifecycle):
            raise TypeError(
                "state_lifecycle must be TrainingStateLifecycle, "
                f"got {type(state_lifecycle).__name__}"
            )
        self.state_lifecycle = state_lifecycle
        if not isinstance(materialize_parameter_gradients, bool):
            raise TypeError("materialize_parameter_gradients must be bool")
        input_gradient_keys = validate_input_gradient_keys(input_gradient_keys)
        self._retain_parameter_gradients = materialize_parameter_gradients
        self.input_gradient_keys = input_gradient_keys
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.zero_stage = ZeroStage(zero_stage)
        self.grad_accum_steps = grad_accum_steps
        self.parallel_mesh = mesh if isinstance(mesh, ParallelMesh) else None
        self.mesh = mesh.mesh if isinstance(mesh, ParallelMesh) else mesh
        checkpoint_devices = (
            tuple(self.mesh.devices.flat)
            if self.mesh is not None
            else (jax.devices()[0],)
        )
        self.checkpoint_mesh = (
            self.mesh
            if self.mesh is not None and "dp" in self.mesh.axis_names
            else Mesh(np.asarray(checkpoint_devices, dtype=object), ("dp",))
        )
        self.checkpoint_sharding = NamedSharding(self.checkpoint_mesh, P())
        self.module: nnx.Module | None = None
        self.optimizer: nnx.Optimizer | None = None
        self._accumulated_gradients: Any | None = None
        self._last_gradients: Any | None = None
        self._gradient_accumulator_jit: Callable[[Any, Any], Any] | None = None
        self.step = 0
        self.micro_step = 0

    @property
    def partition_contract(self) -> ZeroPartitionContract:
        axis_name = (
            self.parallel_mesh.axis_names[0] if self.parallel_mesh is not None else None
        )
        return zero_partition_contract(self.zero_stage, axis_name=axis_name)

    @property
    def accumulated_gradients(self) -> Any | None:
        return self._accumulated_gradients

    @property
    def last_gradients(self) -> Any | None:
        return self._last_gradients

    def initialize(self, *, model_state: Any | None = None) -> None:
        if self.module is not None or self.optimizer is not None:
            raise RuntimeError(
                "ReplicatedNnxTrainEngine.initialize may only be called once"
            )
        if self.zero_stage is not ZeroStage.REPLICATED:
            raise NotImplementedError(
                "ReplicatedNnxTrainEngine requires an explicit distributed execution binding "
                "for ZeRO-1/2/3; use the shard-map stepper directly"
            )
        if self.mesh is not None and any(
            int(size) > 1 for size in self.mesh.shape.values()
        ):
            raise NotImplementedError(
                "ReplicatedNnxTrainEngine eager train_step is single-device; use an explicit "
                "shard-map execution binding for distributed collectives"
            )
        if self.state_lifecycle is TrainingStateLifecycle.CHECKPOINT_RESTORE:
            raise RuntimeError(
                "checkpoint-restore lifecycle cannot initialize fresh values"
            )
        if self.state_lifecycle is TrainingStateLifecycle.MODEL_WEIGHTS:
            if model_state is None:
                raise ValueError(
                    "model_state is required for the model-weights lifecycle"
                )
            values = self.state_definition.initialize_from_model_state(model_state)
        else:
            if model_state is not None:
                raise ValueError(
                    "model_state is only valid for the model-weights lifecycle"
                )
            values = self.state_definition.initialize_values()
        self.state_definition.validate_values(values)
        self.module = values["model"]
        if not isinstance(self.module, nnx.Module):
            raise TypeError(
                "state definition must initialize flax.nnx.Module, "
                f"got {type(self.module).__name__}"
            )
        self.optimizer = values["optimizer"]
        if not isinstance(self.optimizer, nnx.Optimizer):
            raise TypeError(
                "state definition must initialize flax.nnx.Optimizer, "
                f"got {type(self.optimizer).__name__}"
            )
        self.state_definition.validate_compute_module(self.module)

    def train_step(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> TrainStepResult:
        result = self.train_micro_step(inputs, targets)
        if result.optimizer_applied:
            self.apply_accumulated_update()
            return replace(
                result,
                step=self.step,
                micro_step=self.micro_step,
            )
        return result

    def train_micro_step(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> TrainStepResult:
        """Accumulate one gradient micro-step without mutating parameters."""
        if self.module is None or self.optimizer is None:
            raise RuntimeError(
                "ReplicatedNnxTrainEngine.initialize must be called before train_step"
            )
        compute_module = self.state_definition.prepare_compute_module(self.module)
        compute_inputs = self.state_definition.prepare_inputs(inputs)
        compute_loss = self.state_definition.wrap_loss_fn(self.loss_fn)
        if self.input_gradient_keys is None:
            (loss, predictions), (gradients, input_gradients) = (
                compute_loss_and_gradients(
                    module=compute_module,
                    forward_fn=self.forward_fn,
                    loss_fn=compute_loss,
                    inputs=compute_inputs,
                    targets=targets,
                )
            )
        else:
            module_inputs, input_gradient_inputs, _ = partition_input_gradients(
                compute_inputs, self.input_gradient_keys
            )
            (loss, predictions), (gradients, input_gradients) = (
                compute_module_and_input_gradients(
                    module=compute_module,
                    forward_fn=self.forward_fn,
                    loss_fn=compute_loss,
                    input_gradient_inputs=input_gradient_inputs,
                    module_inputs=module_inputs,
                    targets=targets,
                )
            )
        self.state_definition.sync_non_param_state(self.module, compute_module)
        gradients = self.state_definition.cast_gradients(gradients)
        input_gradients = self.state_definition.cast_gradients(input_gradients)
        if self._retain_parameter_gradients:
            self._last_gradients = gradients
        predictions = self.state_definition.cast_predictions(predictions)
        loss = jnp.asarray(loss, dtype=jnp.float32)
        if self._accumulated_gradients is None:
            self._accumulated_gradients = accumulate_gradients(None, gradients)
        else:
            if self._gradient_accumulator_jit is None:
                self._gradient_accumulator_jit = build_donated_gradient_accumulator()
            self._accumulated_gradients = self._gradient_accumulator_jit(
                self._accumulated_gradients, gradients
            )
        self.micro_step += 1
        optimizer_applied = self.micro_step == self.grad_accum_steps
        return TrainStepResult(
            loss=loss,
            predictions=predictions,
            gradients=gradients if self._retain_parameter_gradients else None,
            input_gradients=input_gradients,
            step=self.step,
            micro_step=self.micro_step,
            optimizer_applied=optimizer_applied,
        )

    def apply_accumulated_update(self) -> None:
        """Apply a complete accumulation window and reset its transient state."""
        if self.micro_step != self.grad_accum_steps:
            raise RuntimeError(
                "apply_accumulated_update requires a complete gradient window: "
                f"micro_step={self.micro_step}, grad_accum_steps={self.grad_accum_steps}"
            )
        self._apply_accumulated_update(self.micro_step)

    def finalize_checkpoint_window(self) -> bool:
        if self.micro_step == 0:
            return False
        self._apply_accumulated_update(self.micro_step)
        return True

    def _apply_accumulated_update(self, accumulated_steps: int) -> None:
        if self.module is None or self.optimizer is None:
            raise RuntimeError(
                "ReplicatedNnxTrainEngine.initialize must be called before update"
            )
        if accumulated_steps <= 0 or self.micro_step != accumulated_steps:
            raise RuntimeError("accumulated gradient window is inconsistent")
        if self._accumulated_gradients is None:
            raise RuntimeError("accumulated gradient window has no gradients")
        averaged = mean_accumulated_gradients(
            self._accumulated_gradients,
            accumulated_steps,
        )
        self.optimizer.update(self.module, averaged)
        self._accumulated_gradients = None
        self.micro_step = 0
        self.step += 1

    def capture_checkpoint_state(self) -> dict[str, Any]:
        if self.module is None or self.optimizer is None:
            raise RuntimeError("ReplicatedNnxTrainEngine is not initialized")
        if self.micro_step != 0 or self._accumulated_gradients is not None:
            raise RuntimeError(
                "training checkpoint capture requires a macro-update boundary"
            )
        _, model_state = nnx.split(self.module)
        _, optimizer_state = nnx.split(self.optimizer)
        return {
            "step": jax.device_put(
                jnp.asarray(self.step, dtype=jnp.int32), self.checkpoint_sharding
            ),
            "model_state": state_with_sharding(model_state, self.checkpoint_sharding),
            "optimizer_state": state_with_sharding(
                optimizer_state, self.checkpoint_sharding
            ),
        }

    def checkpoint_layout(self) -> dict[str, Any]:
        if self.zero_stage is not ZeroStage.REPLICATED:
            raise NotImplementedError(
                "ReplicatedNnxTrainEngine checkpoint layout requires replicated execution"
            )
        return self.state_definition.checkpoint_layout(self.checkpoint_sharding)

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if self.state_lifecycle is not TrainingStateLifecycle.CHECKPOINT_RESTORE:
            raise RuntimeError(
                "restore_checkpoint_state requires the checkpoint-restore lifecycle"
            )
        if self.module is not None or self.optimizer is not None:
            raise RuntimeError(
                "restore_checkpoint_state requires an uninitialized engine"
            )
        step = self.state_definition.validate_checkpoint_payload(
            state, context="ReplicatedNnxTrainEngine checkpoint"
        )
        abstract = self.state_definition.abstract_state()
        module = nnx.merge(abstract.module_graphdef, state["model_state"])
        optimizer = nnx.merge(abstract.optimizer_graphdef, state["optimizer_state"])
        self.state_definition.validate_values({"model": module, "optimizer": optimizer})
        self.module = module
        self.optimizer = optimizer
        self.step = step
        self.micro_step = 0
        self._accumulated_gradients = None


__all__ = [
    "ReplicatedNnxTrainEngine",
]
