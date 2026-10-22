from __future__ import annotations

from typing import Any

from fluxway.common.bind import bind_spmd
from fluxway.common.operator_policy import AsyncPolicy, StatePolicy
from fluxway.control_plane.checkpointer import OrbaxCheckpointComponent

from flare.jax.layers.state import NnxModelDefinition
from flare.runtime.fluxway.checkpoint import checkpoint_finalizer_registration
from flare.training.execution import TrainingExecutionPlan
from recipes.recsys.ranking.metrics.eval_metrics import (
    MetricsAggregatorSPMD,
)
from recipes.recsys.ranking.metrics.train_metrics import (
    TrainMetricsAggregatorSPMD,
)
from recipes.recsys.ranking.workers.schedule import ranking_phase_schedule
from recipes.recsys.ranking.workers.train_worker import (
    RecsysBackboneEvalSPMD,
    RecsysBackboneTrainSPMD,
)


def _checkpoint_registration() -> Any:
    return OrbaxCheckpointComponent(
        "module",
        lambda owner, context: owner.capture_checkpoint_state(context),
        lambda owner, state, context: owner.restore_checkpoint_state(state, context),
        lambda owner, context: owner.restore_checkpoint_layout(context),
        output_path="model",
    ).registration()


def _training_checkpoint_registrations() -> tuple[Any, ...]:
    return (
        checkpoint_finalizer_registration(
            "training-finalize",
            lambda owner, context: owner.finalize_checkpoint_state(context),
        ),
        _checkpoint_registration(),
    )


def bind_recsys_ranking_training(
    *,
    sparse_groups: tuple[str, ...],
    side_info_groups: tuple[str, ...],
    resource_profile: Any,
    runtime_env: dict[str, Any],
    dp_size: int,
    pipeline_depth: int,
    execution_plan: TrainingExecutionPlan,
    driver_runtime_env: dict[str, Any] | None = None,
) -> type:
    if dp_size <= 0 or pipeline_depth <= 0:
        raise ValueError("dp_size and pipeline_depth must be positive")
    if not isinstance(execution_plan, TrainingExecutionPlan):
        raise TypeError(
            "execution_plan must be a TrainingExecutionPlan, "
            f"got {type(execution_plan).__name__}"
        )
    batch_input: dict[str, Any] = {}
    materialize_output: dict[str, Any] = {}
    bind_input: dict[str, Any] = {}
    bind_output: dict[str, Any] = {}
    train_input: dict[str, Any] = {}
    train_output: dict[str, Any] = {}
    scatter_input: dict[str, Any] = {}
    scatter_output: dict[str, Any] = {}
    for group in sparse_groups:
        batch_input[f"emb_{group}"] = {
            "type": "dynamic",
            "axis": ["dp"],
            "strategy": "modulo",
        }
        batch_input[f"inv_{group}"] = {"type": "static", "axis": ["dp"]}
        materialize_output[f"sparse_input_{group}"] = {
            "type": "static",
            "axis": [],
        }
        materialize_output[f"padded_inverse_{group}"] = {
            "type": "static",
            "axis": [],
        }
        materialize_output[f"emb_ref_{group}"] = {"type": "static", "axis": []}
        bind_input[f"sparse_input_{group}"] = {"type": "static", "axis": []}
        bind_input[f"padded_inverse_{group}"] = {"type": "static", "axis": []}
        bind_output[f"jax_input_{group}"] = {"type": "static", "axis": []}
        train_input[f"jax_input_{group}"] = {"type": "static", "axis": []}
        train_output[f"grad_jax_{group}"] = {"type": "static", "axis": []}
        scatter_input[f"grad_jax_{group}"] = {"type": "static", "axis": []}
        scatter_input[f"padded_inverse_{group}"] = {"type": "static", "axis": []}
        scatter_input[f"emb_ref_{group}"] = {"type": "static", "axis": []}
        scatter_output[f"grad_{group}"] = {
            "type": "mirror",
            "reference": f"scatter_sparse_grads.emb_ref_{group}",
        }
    for group in side_info_groups:
        batch_input[group] = {"type": "static", "axis": ["dp"]}
        materialize_output[group] = {"type": "static", "axis": []}
        bind_input[group] = {"type": "static", "axis": []}
        bind_output[f"jax_input_{group}"] = {"type": "static", "axis": []}
        train_input[f"jax_input_{group}"] = {"type": "static", "axis": []}
    batch_input.update(
        labels={"type": "static", "axis": ["dp"]},
        samples_valid_count={"type": "static", "axis": []},
    )
    materialize_output.update(
        labels={"type": "static", "axis": []},
        samples_valid_count={"type": "static", "axis": []},
    )
    bind_input.update(
        labels={"type": "static", "axis": []},
        samples_valid_count={"type": "static", "axis": []},
    )
    bind_output.update(
        jax_labels={"type": "static", "axis": []},
        jax_samples_valid_count={"type": "static", "axis": []},
    )
    train_input.update(bind_output)
    train_output.update(
        optimizer_update_window={"type": "static", "axis": []},
        scores={"type": "static", "axis": []},
        loss={"type": "static", "axis": []},
    )
    thread_safe = StatePolicy.thread_safe()

    def pipelined(phase: str, depth: int = pipeline_depth) -> AsyncPolicy:
        return AsyncPolicy.pipelined(
            depth=depth,
            schedule=ranking_phase_schedule(phase),
            state=thread_safe,
        )

    return bind_spmd(
        RecsysBackboneTrainSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(dp_size,),
        runtime_env=runtime_env,
        checkpoint_components=_training_checkpoint_registrations(),
        prepare_sparse_materialize={
            "async_policy": pipelined("prepare_sparse_materialize"),
            "input_sharding": batch_input,
            "output_sharding": materialize_output,
        },
        prepare_jax_bind={
            "async_policy": pipelined("prepare_jax_bind"),
            "input_sharding": bind_input,
            "output_sharding": bind_output,
        },
        train_core={
            "async_policy": pipelined("train_core", 1),
            "input_sharding": train_input,
            "output_sharding": train_output,
        },
        materialize_train_metrics={
            "async_policy": pipelined("materialize_train_metrics"),
            "input_sharding": {
                "scores": {"type": "static", "axis": []},
                "loss": {"type": "static", "axis": []},
                "optimizer_update_window": {"type": "static", "axis": []},
            },
            "output_sharding": {
                "scores": {"type": "static", "axis": ["dp"]},
                "loss": {"type": "static", "axis": ["dp"]},
                "metric_sampled": {"type": "static", "axis": ["dp"]},
            },
        },
        scatter_sparse_grads={
            "async_policy": pipelined("scatter_sparse_grads"),
            "input_sharding": scatter_input,
            "output_sharding": scatter_output,
        },
        update={
            "async_policy": pipelined("update"),
            "input_sharding": {
                "optimizer_update_window": {"type": "static", "axis": []}
            },
        },
        driver_runtime_env=driver_runtime_env,
    )


def bind_recsys_ranking_evaluation(
    *,
    model_definition: NnxModelDefinition,
    sparse_groups: tuple[str, ...],
    side_info_groups: tuple[str, ...],
    resource_profile: Any,
    runtime_env: dict[str, Any],
    dp_size: int,
    pipeline_depth: int = 1,
    driver_runtime_env: dict[str, Any] | None = None,
    checkpoint_components: tuple[Any, ...] = (),
) -> type:
    if dp_size <= 0 or pipeline_depth <= 0:
        raise ValueError("dp_size and pipeline_depth must be positive")
    if not isinstance(model_definition, NnxModelDefinition):
        raise TypeError(
            "model_definition must be NnxModelDefinition, "
            f"got {type(model_definition).__name__}"
        )
    inputs: dict[str, Any] = {}
    for group in sparse_groups:
        inputs[f"emb_{group}"] = {
            "type": "dynamic",
            "axis": ["dp"],
            "strategy": "modulo",
        }
        inputs[f"inv_{group}"] = {"type": "static", "axis": ["dp"]}
    for group in side_info_groups:
        inputs[group] = {"type": "static", "axis": ["dp"]}
    inputs.update(
        labels={"type": "static", "axis": ["dp"]},
        samples_valid_count={"type": "static", "axis": []},
    )
    return bind_spmd(
        RecsysBackboneEvalSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(dp_size,),
        runtime_env=runtime_env,
        checkpoint_components=checkpoint_components,
        step={
            "async_policy": AsyncPolicy.pipelined(
                depth=pipeline_depth,
                state=StatePolicy.thread_safe(),
            ),
            "input_sharding": inputs,
            "output_sharding": {
                "scores": {"type": "static", "axis": ["dp"]},
                "labels": {"type": "static", "axis": ["dp"]},
            },
        },
        driver_runtime_env=driver_runtime_env,
    )


def bind_recsys_ranking_train_metrics(
    *,
    input_sharding: dict[str, Any],
    resource_profile: Any,
    runtime_env: dict[str, Any],
    max_inflight_steps: int,
    driver_runtime_env: dict[str, Any] | None = None,
) -> type:
    if max_inflight_steps <= 0:
        raise ValueError("max_inflight_steps must be positive")
    return bind_spmd(
        TrainMetricsAggregatorSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(1,),
        runtime_env=runtime_env,
        checkpoint_components=(),
        step={
            "async_policy": AsyncPolicy.pipelined(depth=max_inflight_steps),
            "input_sharding": input_sharding,
            "output_sharding": {},
        },
        driver_runtime_env=driver_runtime_env,
    )


def bind_recsys_ranking_metrics(
    *,
    input_sharding: dict[str, Any],
    resource_profile: Any,
    runtime_env: dict[str, Any],
    driver_runtime_env: dict[str, Any] | None = None,
) -> type:
    return bind_spmd(
        MetricsAggregatorSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(1,),
        runtime_env=runtime_env,
        checkpoint_components=(),
        step={
            "input_sharding": input_sharding,
            "output_sharding": {},
        },
        driver_runtime_env=driver_runtime_env,
    )


__all__ = [
    "bind_recsys_ranking_evaluation",
    "bind_recsys_ranking_metrics",
    "bind_recsys_ranking_train_metrics",
    "bind_recsys_ranking_training",
]
