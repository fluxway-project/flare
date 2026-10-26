from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np
from flax import nnx
from fluxway.common.decorators import requires_worker_process
from fluxway.common.index_descriptor import ContinuousIndex
from jax.sharding import PartitionSpec as P

from flare.jax.layers.state import NnxModelDefinition
from flare.jax.mesh import ParallelMesh
from flare.jax.training.engines.contracts import TrainStepResult
from flare.jax.training.engines.distributed import initialize_distributed_jax
from flare.jax.training.engines.replicated import ReplicatedNnxTrainEngine
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.kernels import (
    release_jax_leases,
    to_host_array,
    to_jax_array,
)
from flare.kernels.conversion import to_jax_array_with_lease
from flare.ops import create_sparse_embedding_operator
from flare.runtime.diagnostics.state_audit import fingerprint_state
from flare.runtime.fluxway.transport import make_transport_tensor, transport_payload
from flare.runtime.fluxway.workers.base import require_operator_step
from flare.runtime.sparse_precision import (
    SPARSE_NATIVE_GRADIENT_DTYPE,
    require_finite_sparse_values,
    require_sparse_tensor_dtype,
)
from flare.training.execution import (
    DistributedTrainStepExecutor,
    OptimizerUpdateWindow,
    ReplicatedTrainStepExecutor,
    TrainingExecutionPlan,
    TrainStepExecutor,
)
from flare.training.state import TrainingStateLifecycle
from flare.training.zero.contracts import ZeroStage
from recipes.recsys.ranking.metrics.execution_trace import RecsysExecutionTracer
from recipes.recsys.ranking.schema.materialization import (
    materialize_recsys_sparse_inputs,
)

logger = logging.getLogger(__name__)


def normalize_sparse_group_widths(
    sparse_group_widths: Mapping[str, tuple[int, int]],
    sparse_groups: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    missing = [group for group in sparse_groups if group not in sparse_group_widths]
    if missing:
        raise ValueError(f"Missing sparse group widths: {missing}")
    normalized: dict[str, tuple[int, int]] = {}
    for group in sparse_groups:
        widths = tuple(value for value in sparse_group_widths[group])
        if len(widths) != 2:
            raise ValueError(
                f"Sparse group width for {group} must be length 2, got {widths}"
            )
        if any(value <= 0 for value in widths):
            raise ValueError(
                f"Sparse group width for {group} must be positive, got {widths}"
            )
        normalized[group] = (widths[0], widths[1])
    return normalized


@requires_worker_process
class RecsysBackboneTrainSPMD:
    rank: int
    world_size: int
    local_rank: int | None
    local_world_size: int | None
    ctx: Any
    rank0_address: str

    def __init__(
        self,
        state_definition: NnxTrainingStateDefinition,
        sparse_groups: tuple[str, ...],
        side_info_groups: tuple[str, ...],
        sparse_group_widths: dict[str, tuple[int, int]],
        sparse_tensor_backend: str,
        forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
        loss_fn: Callable[[Any, Mapping[str, Any]], Any],
        execution_plan: TrainingExecutionPlan,
        grad_accum_steps: int = 1,
        metric_sample_interval_steps: int = 1,
        target_step: int | None = None,
        train_eval_metrics_enabled: bool = False,
        execution_tracer: RecsysExecutionTracer | None = None,
        validate_tensor_values: bool = False,
        distributed_executor_factory: Callable[..., TrainStepExecutor] | None = None,
        state_lifecycle: TrainingStateLifecycle = TrainingStateLifecycle.FRESH_INITIALIZE,
        model_state: Any | None = None,
    ) -> None:
        if not isinstance(execution_plan, TrainingExecutionPlan):
            raise TypeError(
                "execution_plan must be a TrainingExecutionPlan, "
                f"got {type(execution_plan).__name__}"
            )
        if execution_tracer is not None:
            execution_tracer.prepare()
        self.execution_plan = execution_plan
        if (
            self.execution_plan.zero.stage is not ZeroStage.REPLICATED
            and self.world_size == 1
        ):
            raise ValueError(
                f"ZeRO stage {self.execution_plan.zero.stage.name} requires world_size > 1, got {self.world_size}"
            )
        if self.world_size > 1:
            initialize_distributed_jax(
                rank=self.rank,
                world_size=self.world_size,
                local_rank=self._require_local_rank(),
                local_world_size=self.local_world_size,
                rank0_address=self.rank0_address,
            )
            logger.info(
                "jax.distributed.ready rank=%s world_size=%s process_index=%s process_count=%s zero_stage=%s",
                self.rank,
                self.world_size,
                jax.process_index(),
                jax.process_count(),
                execution_plan.zero.stage.name,
            )
        if not isinstance(state_definition, NnxTrainingStateDefinition):
            raise TypeError(
                "state_definition must be NnxTrainingStateDefinition, "
                f"got {type(state_definition).__name__}"
            )
        self.state_definition = state_definition
        self.sparse_groups = tuple(sparse_groups)
        self.side_info_groups = tuple(side_info_groups)
        self.sparse_embedding_dtype = (
            state_definition.precision_policy.sparse_storage_dtype_name
        )
        self.sparse_gradient_transport_dtype = (
            state_definition.precision_policy.sparse_gradient_transport_dtype_name
        )
        self.sparse_gradient_dtype = (
            state_definition.precision_policy.sparse_gradient_dtype_name
        )
        self.sparse_group_widths = normalize_sparse_group_widths(
            sparse_group_widths, self.sparse_groups
        )
        self.sparse_op = create_sparse_embedding_operator(
            sparse_tensor_backend,
            value_dtype=self.sparse_embedding_dtype,
            gradient_dtype=self.sparse_gradient_dtype,
            empty_cache=(sparse_tensor_backend == "cupy"),
        )
        if grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {grad_accum_steps}"
            )
        if metric_sample_interval_steps <= 0:
            raise ValueError(
                f"metric_sample_interval_steps must be positive, got {metric_sample_interval_steps}"
            )
        if target_step is not None and target_step <= 0:
            raise ValueError(f"target_step must be positive, got {target_step}")
        self.grad_accum_steps = grad_accum_steps
        self.metric_sample_interval_steps = metric_sample_interval_steps
        self.target_step = None if target_step is None else target_step
        self.train_eval_metrics_enabled = train_eval_metrics_enabled
        if execution_tracer is not None:
            self.execution_tracer = execution_tracer
            self._execution_event_sequence = 0
            if execution_tracer.include_values:
                self._trace_values = True
        if not isinstance(validate_tensor_values, bool):
            raise TypeError("validate_tensor_values must be bool")
        if validate_tensor_values:
            self.validate_tensor_values = True
        self.micro_step = 0
        self._prepared_execution_for_next_batch = False
        if execution_tracer is not None and execution_tracer.include_values:
            self._last_module_before: Mapping[str, Any] | None = None
        if self.execution_plan.accumulation.micro_steps != self.grad_accum_steps:
            raise ValueError("execution_plan accumulation must equal grad_accum_steps")
        self._engine: ReplicatedNnxTrainEngine | None = None
        self._executor: TrainStepExecutor
        self._last_result: TrainStepResult | None = None
        self._last_update_window: OptimizerUpdateWindow | None = None
        self._last_update_window_payload: np.ndarray | None = None
        self._pending_jax_leases: list[tuple[Any, ...]] = []
        self._forward_fn = forward_fn
        self._loss_fn = loss_fn
        if (
            self.execution_plan.zero.stage is ZeroStage.REPLICATED
            and self.world_size == 1
        ):
            self.configure_training(
                forward_fn,
                loss_fn,
                state_lifecycle=state_lifecycle,
                model_state=model_state,
            )
            if self._engine is None:
                raise RuntimeError("replicated training engine was not initialized")
            self._executor = ReplicatedTrainStepExecutor(
                self._engine, self.execution_plan
            )
        else:
            if distributed_executor_factory is None:
                raise ValueError(
                    "distributed_executor_factory is required for multi-rank execution "
                    f"(zero_stage={self.execution_plan.zero.stage.name}, world_size={self.world_size})"
                )
            self._executor = distributed_executor_factory(
                execution_plan=self.execution_plan,
                state_definition=state_definition,
                state_lifecycle=state_lifecycle,
                model_state=model_state,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
                rank=self.rank,
                world_size=self.world_size,
                local_rank=self._require_local_rank(),
                local_world_size=self.local_world_size,
                rank0_address=self.rank0_address,
                input_gradient_keys=self.sparse_groups,
                target_shardings={
                    "labels": P("dp"),
                    "samples_valid_count": P(),
                    "global_row_start": P(),
                    "world_size": P(),
                },
                materialize_parameter_gradients=getattr(self, "_trace_values", False),
            )
            if not isinstance(self._executor, DistributedTrainStepExecutor):
                raise TypeError(
                    "distributed_executor_factory must return DistributedTrainStepExecutor"
                )

    def _require_local_rank(self) -> int:
        if self.local_rank is None:
            raise RuntimeError(
                "Fluxway did not provide local_rank before distributed training initialization"
            )
        return self.local_rank

    def _record_execution_phase(
        self, stage: str, *, payload: Mapping[str, Any] | None = None
    ) -> None:
        tracer = getattr(self, "execution_tracer", None)
        if tracer is None:
            return
        sequence = getattr(self, "_execution_event_sequence", 0)
        self._execution_event_sequence += 1
        tracer.record_event(
            stage=stage,
            step=self.micro_step,
            spmd_id="RecsysBackboneTrain",
            rank=self.rank,
            group="operator_execution",
            payload={"sequence": sequence, **(payload or {})},
        )

    def prepare_sparse_materialize(self, **kwargs: Any) -> dict[str, Any]:
        return materialize_recsys_sparse_inputs(
            kwargs=kwargs,
            sparse_groups=self.sparse_groups,
            side_info_groups=self.side_info_groups,
            sparse_group_widths=self.sparse_group_widths,
            sparse_op=self.sparse_op,
            execution_tracer=getattr(self, "execution_tracer", None),
            world_size=self.world_size,
            rank=self.rank,
            spmd_id="RecsysBackboneTrain",
            operator_owner=self,
            validate_tensor_values=getattr(self, "validate_tensor_values", False),
        )

    def prepare_jax_bind(self, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        leases = []

        def bind(value: Any) -> Any:
            # The consumer lease pins the transport payload for the duration of
            # this operator.  Preserve the backend's zero-copy import policy;
            # preparation must not duplicate every fixed-shape batch.
            array, lease = to_jax_array_with_lease(value)
            if lease is not None:
                leases.append(lease)
            return array

        try:
            for group in self.sparse_groups:
                name = f"sparse_input_{group}"
                if name not in kwargs:
                    raise KeyError(f"prepare_jax_bind missing {name!r}")
                result[f"jax_input_{group}"] = bind(kwargs[name])
            for group in self.side_info_groups:
                if group not in kwargs:
                    raise KeyError(
                        f"prepare_jax_bind missing side-info group {group!r}"
                    )
                result[f"jax_input_{group}"] = bind(kwargs[group])
            result["jax_labels"] = bind(kwargs["labels"])
            result["jax_samples_valid_count"] = bind(kwargs["samples_valid_count"])
            prepare_inputs = {
                group: result[f"jax_input_{group}"]
                for group in (*self.sparse_groups, *self.side_info_groups)
            }
            prepare_targets = {
                "labels": result["jax_labels"],
                "samples_valid_count": result["jax_samples_valid_count"],
                "global_row_start": self.rank * result["jax_labels"].shape[0],
                "world_size": self.world_size,
            }
            self._record_execution_phase(
                "operator_prepare_begin", payload={"phase": "compile"}
            )
            self._executor.prepare_execution(prepare_inputs, prepare_targets)
            self._record_execution_phase(
                "operator_prepare_end", payload={"phase": "compile"}
            )
            self._prepared_execution_for_next_batch = True
            self._pending_jax_leases.append(tuple(leases))
            return result
        except BaseException:
            release_jax_leases(leases)
            raise

    def configure_training(
        self,
        forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
        loss_fn: Callable[[Any, Mapping[str, Any]], Any],
        *,
        mesh: Any | None = None,
        state_lifecycle: TrainingStateLifecycle = TrainingStateLifecycle.FRESH_INITIALIZE,
        model_state: Any | None = None,
    ) -> None:
        if self._engine is not None:
            raise RuntimeError("RecsysBackboneTrainSPMD training is already configured")
        self._engine = ReplicatedNnxTrainEngine(
            self.state_definition,
            forward_fn,
            loss_fn,
            zero_stage=self.execution_plan.zero.stage,
            grad_accum_steps=self.grad_accum_steps,
            mesh=mesh,
            state_lifecycle=state_lifecycle,
            materialize_parameter_gradients=getattr(self, "_trace_values", False),
            input_gradient_keys=self.sparse_groups,
        )
        if state_lifecycle is not TrainingStateLifecycle.CHECKPOINT_RESTORE:
            self._engine.initialize(model_state=model_state)

    def _require_engine(self) -> ReplicatedNnxTrainEngine:
        if self._engine is None:
            raise RuntimeError(
                "RecsysBackboneTrainSPMD.configure_training must be called before training"
            )
        if self._engine.module is None or self._engine.optimizer is None:
            raise RuntimeError(
                "training state is not materialized; choose fresh initialization, "
                "model-weight initialization, or checkpoint restore before train_core"
            )
        return self._engine

    def train_core(self, **kwargs: Any) -> dict[str, Any]:
        leases = self._pending_jax_leases.pop(0)
        try:
            inputs = {
                group: to_jax_array(kwargs[f"jax_input_{group}"])
                for group in (*self.sparse_groups, *self.side_info_groups)
            }
            labels = to_jax_array(kwargs["jax_labels"])
            samples_valid_count = to_jax_array(kwargs["jax_samples_valid_count"])
            targets = {
                "labels": labels,
                "samples_valid_count": samples_valid_count,
                "global_row_start": self.rank * labels.shape[0],
                "world_size": self.world_size,
            }
            preparation_already_recorded = self._prepared_execution_for_next_batch
            if not preparation_already_recorded:
                self._record_execution_phase(
                    "operator_prepare_begin", payload={"phase": "compile"}
                )
                self._executor.prepare_execution(inputs, targets)
                self._record_execution_phase(
                    "operator_prepare_end", payload={"phase": "compile"}
                )
            self._prepared_execution_for_next_batch = False
            if getattr(self, "_trace_values", False):
                self._last_module_before = self._executor.capture_module_fingerprint()
            self._record_execution_phase(
                "operator_invocation_begin", payload={"phase": "execute"}
            )
            result, update_window = self._executor.micro_step(inputs, targets)
            self._record_execution_phase(
                "operator_invocation_end", payload={"phase": "execute"}
            )
        finally:
            release_jax_leases(leases)
        self._last_result = result
        self._last_update_window = update_window
        self.micro_step = result.micro_step
        self._last_update_window_payload = np.asarray(
            update_window.encode(), dtype=np.int64
        )
        predictions = result.predictions
        sparse_input_gradients = result.input_gradients
        if not isinstance(predictions, Mapping) or "logits" not in predictions:
            raise TypeError(
                "module forward function must return a mapping containing 'logits'"
            )
        output = {
            "optimizer_update_window": self._last_update_window_payload.copy(),
            "scores": jax.nn.sigmoid(predictions["logits"]),
            "loss": result.loss,
        }
        for group in self.sparse_groups:
            gradient = sparse_input_gradients.get(group)
            if gradient is None:
                raise RuntimeError(
                    f"module input gradient is missing sparse group {group!r}"
                )
            # Keep the two divisors at their owners: the distributed JAX engine
            # applies the sample-mean DP factor, while the sparse owner divides
            # its keyed sum by the actual commit-window length. Scaling here by
            # configured accumulation steps would double-divide full windows
            # and under-scale partial windows; ID frequency is not either factor.
            output[f"grad_jax_{group}"] = gradient
        return output

    def scatter_sparse_grads(self, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for group in self.sparse_groups:
            gradient = kwargs.get(f"grad_jax_{group}")
            inverse_name = f"padded_inverse_{group}"
            embedding_name = f"emb_ref_{group}"
            operator_step = require_operator_step(
                self, "RecsysBackboneTrain.scatter_sparse_grads"
            )
            gradient_array = self.sparse_op.local_array(transport_payload(gradient))
            if (
                np.dtype(gradient_array.dtype).name
                != self.sparse_gradient_transport_dtype
            ):
                gradient_array = gradient_array.astype(
                    self.sparse_gradient_transport_dtype, copy=False
                )
            if getattr(self, "validate_tensor_values", False):
                gradient_array = require_finite_sparse_values(
                    gradient_array,
                    self.sparse_gradient_transport_dtype,
                    boundary=f"model_gradient_input:{group}",
                )
            gradient_array = require_sparse_tensor_dtype(
                gradient_array,
                self.sparse_gradient_transport_dtype,
                boundary=f"model_gradient_input:{group}",
            )
            emb_ref = kwargs[embedding_name]
            embedding_ref = self.sparse_op.local_array(transport_payload(emb_ref))
            inverse_value = kwargs[inverse_name]
            emb_dim = int(embedding_ref.shape[-1])
            grad = self.sparse_op.scatter_pooled(
                int(embedding_ref.shape[0]),
                emb_dim,
                self.sparse_op.inverse(transport_payload(inverse_value)),
                int(embedding_ref.shape[0]),
                gradient_array,
                self.sparse_group_widths[group],
            )
            if getattr(self, "validate_tensor_values", False):
                grad = require_finite_sparse_values(
                    grad,
                    self.sparse_op.gradient_dtype,
                    boundary=f"model_gradient_scatter:{group}",
                )
            grad = require_sparse_tensor_dtype(
                grad,
                self.sparse_op.gradient_dtype,
                boundary=f"model_gradient_scatter:{group}",
            )
            result[f"grad_{group}"] = make_transport_tensor(
                grad, source_step=operator_step
            )
            tracer = getattr(self, "execution_tracer", None)
            if tracer is not None:
                tracer.module_gradient(
                    group=group,
                    step=operator_step,
                    spmd_id="RecsysBackboneTrain",
                    rank=self.rank,
                    gradient=grad,
                    inverse=transport_payload(inverse_value),
                    embedding_tensor=emb_ref,
                )
        return result

    def materialize_train_metrics(self, **kwargs: Any) -> dict[str, Any]:
        if "optimizer_update_window" not in kwargs:
            raise KeyError("materialize_train_metrics requires optimizer_update_window")
        if self.train_eval_metrics_enabled:
            current_step = require_operator_step(
                self, "RecsysBackboneTrain.materialize_train_metrics"
            )
            should_sample = (
                current_step + 1
            ) % self.metric_sample_interval_steps == 0 or (
                self.target_step is not None and current_step + 1 >= self.target_step
            )
        else:
            should_sample = False
        if not should_sample:
            return {
                "scores": make_transport_tensor(
                    np.zeros((1, 1), dtype=np.float32),
                    global_shape=(self.world_size, 1),
                ),
                "loss": make_transport_tensor(
                    np.zeros((1,), dtype=np.float32), global_shape=(self.world_size,)
                ),
                "metric_sampled": make_transport_tensor(
                    np.zeros((1,), dtype=np.int32), global_shape=(self.world_size,)
                ),
            }
        scores_value = transport_payload(kwargs["scores"])
        loss_value = transport_payload(kwargs["loss"])
        scores = to_host_array(
            self.sparse_op.local_array(scores_value), dtype=np.float32
        )
        loss = to_host_array(
            self.sparse_op.local_array(loss_value), dtype=np.float32
        ).reshape(-1)
        if getattr(self, "execution_tracer", None) is not None and (
            scores.ndim != 2 or loss.size != 1
        ):
            raise ValueError(
                "sampled train metrics require rank-2 scores and scalar loss"
            )
        return {
            "scores": make_transport_tensor(
                scores,
                global_shape=(scores.shape[0] * self.world_size, scores.shape[1]),
            ),
            "loss": make_transport_tensor(loss, global_shape=(self.world_size,)),
            "metric_sampled": make_transport_tensor(
                np.ones((1,), dtype=np.int32), global_shape=(self.world_size,)
            ),
        }

    def update(self, optimizer_update_window: Any) -> bool:
        result = self._last_result
        if result is None:
            raise RuntimeError(
                "RecsysBackboneTrainSPMD.update requires a preceding train_core call"
            )
        update_window = self._require_matching_update_window(optimizer_update_window)
        audit_before_update = (
            self._executor.capture_audit_state()
            if getattr(self, "_trace_values", False)
            else None
        )
        applied = self._executor.apply_update(update_window)
        if applied:
            self.micro_step = self._executor.current_micro_step()
        tracer = getattr(self, "execution_tracer", None)
        if tracer is not None:
            module_after = None
            if getattr(self, "_trace_values", False):
                module_before = getattr(self, "_last_module_before", None)
                if module_before is None or audit_before_update is None:
                    raise RuntimeError("audit state is missing for train_core result")
                module_after = (
                    self._executor.capture_module_fingerprint()
                    if applied
                    else module_before
                )
            tracer.module_update(
                step=require_operator_step(self, "RecsysBackboneTrain.update"),
                spmd_id="RecsysBackboneTrain",
                rank=self.rank,
                micro_step=update_window.micro_step,
                grad_accum_steps=self.grad_accum_steps,
                optimizer_applied=applied,
                module_before=getattr(self, "_last_module_before", None),
                module_after=module_after,
                accumulated_gradients=(
                    audit_before_update["accumulated_gradients"]
                    if audit_before_update is not None
                    else None
                ),
                raw_gradients=(
                    audit_before_update["raw_gradients"]
                    if audit_before_update is not None
                    else None
                ),
                reduced_gradients=(
                    audit_before_update["reduced_gradients"]
                    if audit_before_update is not None
                    else None
                ),
            )
        self._last_result = None
        self._last_update_window_payload = None
        self._last_update_window = None
        if hasattr(self, "_last_module_before"):
            del self._last_module_before
        return result.optimizer_applied

    def _require_matching_update_window(self, value: Any) -> OptimizerUpdateWindow:
        if self._last_update_window is None:
            raise RuntimeError("update window requires a preceding train_core")
        payload = getattr(value, "flat_values", value)
        window = OptimizerUpdateWindow.decode(
            np.asarray(payload, dtype=np.int64).reshape(-1)
        )
        if window != self._last_update_window:
            raise ValueError(
                "RecsysBackboneTrainSPMD optimizer update window does not match train_core"
            )
        return window

    def capture_checkpoint_state(self, context: Any | None = None) -> dict[str, Any]:
        return dict(self._executor.capture_checkpoint_state())

    def finalize_checkpoint_state(self, _context: Any) -> None:
        self._executor.finalize_checkpoint_window()
        self.micro_step = self._executor.current_micro_step()
        self._last_result = None
        self._last_update_window_payload = None
        self._last_update_window = None
        if hasattr(self, "_last_module_before"):
            del self._last_module_before

    def restore_checkpoint_layout(self, context: Any | None = None) -> dict[str, Any]:
        return dict(self._executor.checkpoint_layout())

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        context: Any | None = None,
    ) -> None:
        self._executor.restore_checkpoint_state(state)
        self.micro_step = self._executor.current_micro_step()

    def audit_state_snapshot(self) -> dict[str, Any]:
        audit = self._executor.capture_audit_state()
        return {
            "kind": "recsys_ranking_train",
            "rank": self.rank,
            "world_size": self.world_size,
            "micro_step": audit["micro_step"],
            "grad_accum_steps": audit["grad_accum_steps"],
            "module": audit["module"],
            "optimizer": audit["optimizer"],
            "accumulated_gradients": audit["accumulated_gradients"],
        }


@requires_worker_process
class RecsysBackboneEvalSPMD:
    rank: int
    world_size: int

    def __init__(
        self,
        model_definition: NnxModelDefinition,
        sparse_groups: tuple[str, ...],
        side_info_groups: tuple[str, ...],
        RankingOutput: type,
        sparse_group_widths: dict[str, tuple[int, int]],
        sparse_tensor_backend: str,
        forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
        weight_source: Any,
        sparse_embedding_dtype: str,
        execution_tracer: RecsysExecutionTracer | None = None,
        validate_tensor_values: bool = False,
    ) -> None:
        if not isinstance(model_definition, NnxModelDefinition):
            raise TypeError("evaluation model_definition must be NnxModelDefinition")
        if execution_tracer is not None:
            execution_tracer.prepare()
        self.model_definition = model_definition
        self.sparse_embedding_dtype = np.dtype(sparse_embedding_dtype).name
        self.sparse_gradient_dtype = SPARSE_NATIVE_GRADIENT_DTYPE
        self.sparse_groups = tuple(sparse_groups)
        self.side_info_groups = tuple(side_info_groups)
        self.RankingOutput = RankingOutput
        self.sparse_group_widths = normalize_sparse_group_widths(
            sparse_group_widths, self.sparse_groups
        )
        if execution_tracer is not None:
            self.execution_tracer = execution_tracer
        if not isinstance(validate_tensor_values, bool):
            raise TypeError("validate_tensor_values must be bool")
        if validate_tensor_values:
            self.validate_tensor_values = True
        self.sparse_op = create_sparse_embedding_operator(
            sparse_tensor_backend,
            value_dtype=self.sparse_embedding_dtype,
            gradient_dtype=self.sparse_gradient_dtype,
        )
        if not callable(forward_fn):
            raise TypeError("forward_fn must be callable")

        @nnx.jit
        def forward_scores(module: nnx.Module, inputs: Mapping[str, Any]) -> Any:
            predictions = forward_fn(module, inputs)
            if not isinstance(predictions, Mapping) or "logits" not in predictions:
                raise TypeError(
                    "module evaluation forward must return mapping with 'logits'"
                )
            return jax.nn.sigmoid(predictions["logits"])

        self._forward_scores = forward_scores
        if not hasattr(weight_source, "model") or not hasattr(weight_source, "sparse"):
            raise TypeError("evaluation requires model and sparse inference weights")
        device_mesh = getattr(weight_source, "device_mesh", None)
        if device_mesh is None:
            raise TypeError("evaluation weight source must carry a Fluxway DeviceMesh")
        local_devices = tuple(jax.local_devices())
        if not local_devices:
            raise RuntimeError("evaluation requires at least one local JAX device")
        materialized_mesh = ParallelMesh.from_dp(
            size=1,
            devices=(local_devices[0],),
            axis_name="dp",
        ).mesh
        module_state = weight_source.model.load(
            self.model_definition, materialized_mesh
        )
        self.module = self.model_definition.materialize(module_state)
        self.target_device = materialized_mesh.devices.flat[0]

    def step(self, **kwargs: Any) -> Any:
        prepared = materialize_recsys_sparse_inputs(
            kwargs=kwargs,
            sparse_groups=self.sparse_groups,
            side_info_groups=self.side_info_groups,
            sparse_group_widths=self.sparse_group_widths,
            sparse_op=self.sparse_op,
            execution_tracer=getattr(self, "execution_tracer", None),
            world_size=self.world_size,
            rank=self.rank,
            spmd_id="RecsysBackboneEval",
            operator_owner=self,
            validate_tensor_values=getattr(self, "validate_tensor_values", False),
        )
        leases = []
        try:
            inputs = {}
            for group in self.sparse_groups:
                array, lease = to_jax_array_with_lease(
                    prepared[f"sparse_input_{group}"]
                )
                if array.devices() != {self.target_device}:
                    array = jax.device_put(array, self.target_device)
                inputs[group] = array
                if lease is not None:
                    leases.append(lease)
            for group in self.side_info_groups:
                array, lease = to_jax_array_with_lease(prepared[group])
                if array.devices() != {self.target_device}:
                    array = jax.device_put(array, self.target_device)
                inputs[group] = array
                if lease is not None:
                    leases.append(lease)
            scores = np.asarray(
                jax.device_get(self._forward_scores(self.module, inputs)),
                dtype=np.float32,
            )
        finally:
            release_jax_leases(leases)
        labels = to_host_array(transport_payload(kwargs["labels"]), dtype=np.int32)
        if getattr(self, "execution_tracer", None) is not None and (
            scores.ndim != 2 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]
        ):
            raise ValueError("evaluation scores and labels must be row-aligned")
        origins = ContinuousIndex.chunk(self.rank * labels.shape[0], labels.shape[0])
        global_rows = labels.shape[0] * self.world_size
        return self.RankingOutput(
            scores=make_transport_tensor(
                scores,
                global_shape=(global_rows, scores.shape[1]),
                origin_indices=origins,
            ),
            labels=make_transport_tensor(
                flat_values=labels,
                global_shape=(global_rows,),
                origin_indices=origins,
            ),
        )

    def audit_state_snapshot(self) -> dict[str, Any]:
        return {
            "kind": "recsys_ranking_eval",
            "rank": self.rank,
            "world_size": self.world_size,
            "module": fingerprint_state(nnx.state(self.module)),
        }


__all__ = [
    "RecsysBackboneEvalSPMD",
    "RecsysBackboneTrainSPMD",
    "normalize_sparse_group_widths",
]
