from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flare.jax.layers.state import NnxModelDefinition
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.training.execution import TrainingExecutionPlan
from recipes.recsys.ranking.graph.bindings import (
    bind_recsys_ranking_evaluation,
    bind_recsys_ranking_training,
)


@dataclass(frozen=True)
class RecsysBackboneBindings:
    """Optional ranking preset for native Fluxway graph assembly."""

    training_spmd: type
    evaluation_spmd: type
    sparse_groups: tuple[str, ...]
    side_info_groups: tuple[str, ...]

    def training_static_args(
        self,
        *,
        state_definition: NnxTrainingStateDefinition,
        sparse_group_widths: Mapping[str, tuple[int, int]],
        sparse_tensor_backend: str,
        forward_fn: Any,
        loss_fn: Any,
        execution_plan: TrainingExecutionPlan,
        grad_accum_steps: int = 1,
        metric_sample_interval_steps: int = 1,
        target_step: int | None = None,
        train_eval_metrics_enabled: bool = False,
        execution_tracer: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "state_definition": state_definition,
            "sparse_groups": self.sparse_groups,
            "side_info_groups": self.side_info_groups,
            "sparse_group_widths": dict(sparse_group_widths),
            "sparse_tensor_backend": sparse_tensor_backend,
            "forward_fn": forward_fn,
            "loss_fn": loss_fn,
            "execution_plan": execution_plan,
            "grad_accum_steps": grad_accum_steps,
            "metric_sample_interval_steps": metric_sample_interval_steps,
            "target_step": target_step,
            "train_eval_metrics_enabled": train_eval_metrics_enabled,
            "execution_tracer": execution_tracer,
        }

    def evaluation_static_args(
        self,
        *,
        model_definition: NnxModelDefinition,
        ranking_output: type,
        sparse_group_widths: Mapping[str, tuple[int, int]],
        sparse_tensor_backend: str,
        forward_fn: Any,
        weight_source: Any,
        sparse_embedding_dtype: str,
        execution_tracer: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "model_definition": model_definition,
            "sparse_groups": self.sparse_groups,
            "side_info_groups": self.side_info_groups,
            "RankingOutput": ranking_output,
            "sparse_group_widths": dict(sparse_group_widths),
            "sparse_tensor_backend": sparse_tensor_backend,
            "forward_fn": forward_fn,
            "weight_source": weight_source,
            "sparse_embedding_dtype": sparse_embedding_dtype,
            "execution_tracer": execution_tracer,
        }

    def input_static_args(
        self,
        *,
        input_node_output: type,
        key_dtype: str,
        execution_tracer: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "sparse_groups": self.sparse_groups,
            "InputNodeOutput": input_node_output,
            "key_dtype": key_dtype,
            "execution_tracer": execution_tracer,
        }


def bind_recsys_ranking_components(
    *,
    evaluation_model_definition: NnxModelDefinition,
    sparse_groups: tuple[str, ...],
    side_info_groups: tuple[str, ...],
    resource_profile: Any,
    runtime_env: dict[str, Any],
    dp_size: int,
    pipeline_depth: int,
    execution_plan: TrainingExecutionPlan,
    driver_runtime_env: dict[str, Any] | None = None,
) -> RecsysBackboneBindings:
    """Return optional ranking bindings; the caller owns every Fluxway graph."""

    if not isinstance(evaluation_model_definition, NnxModelDefinition):
        raise TypeError(
            "evaluation_model_definition must be NnxModelDefinition, "
            f"got {type(evaluation_model_definition).__name__}"
        )

    return RecsysBackboneBindings(
        training_spmd=bind_recsys_ranking_training(
            sparse_groups=sparse_groups,
            side_info_groups=side_info_groups,
            resource_profile=resource_profile,
            runtime_env=runtime_env,
            dp_size=dp_size,
            pipeline_depth=pipeline_depth,
            execution_plan=execution_plan,
            driver_runtime_env=driver_runtime_env,
        ),
        evaluation_spmd=bind_recsys_ranking_evaluation(
            model_definition=evaluation_model_definition,
            sparse_groups=sparse_groups,
            side_info_groups=side_info_groups,
            resource_profile=resource_profile,
            runtime_env=runtime_env,
            dp_size=dp_size,
            pipeline_depth=pipeline_depth,
            driver_runtime_env=driver_runtime_env,
        ),
        sparse_groups=tuple(sparse_groups),
        side_info_groups=tuple(side_info_groups),
    )


__all__ = ["RecsysBackboneBindings", "bind_recsys_ranking_components"]
