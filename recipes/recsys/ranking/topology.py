from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from fluxway.common.logical_partition_binding import LogicalPartitionBinding
from fluxway.common.mesh import DeviceMesh
from fluxway.control_plane.graph.functional import FunctionalGraph, NodeProxy
from sparse_embedding.checkpoint import PartitionSpec

from flare.mesh import (
    SparsePartitionMapping,
    bind_sparse_worker_meshes,
)
from flare.runtime.fluxway.placement import build_affinity_placement
from flare.runtime.sparse_restore import build_sparse_restore_plans
from recipes.recsys.ranking.schema.sparse_features import parse_sparse_group_name


@dataclass(frozen=True, slots=True)
class RankingTrainingHandles:
    training: Mapping[str, Any]
    metrics: Any | None
    training_node: NodeProxy


@dataclass(frozen=True, slots=True)
class RankingTrainingAssemblyConfig:
    sparse_groups: tuple[str, ...]
    side_info_groups: tuple[str, ...]
    training_spmd: type
    training_mesh: DeviceMesh
    training_static_args: Mapping[str, Any]
    metrics_spmd: type | None
    metrics_mesh: DeviceMesh | None
    metrics_static_args: Mapping[str, Any] | None
    compute_spmd_name: str
    metrics_spmd_name: str


def bind_sparse_embedding_restore_plans(
    *,
    partition_mappings: Mapping[str, SparsePartitionMapping],
    restore_plans_by_partition: Mapping[str, Mapping[int, bytes]],
) -> Mapping[str, Mapping[int, bytes]]:
    if set(partition_mappings) != set(restore_plans_by_partition):
        raise ValueError(
            "partition mappings and restore plans must cover the same groups"
        )
    bound: dict[str, Mapping[int, bytes]] = {}
    for group, mapping in partition_mappings.items():
        if not isinstance(mapping, SparsePartitionMapping):
            raise TypeError(
                f"partition mapping for {group!r} must be SparsePartitionMapping"
            )
        restore_plans = restore_plans_by_partition[group]
        expected = set(range(mapping.partition_count))
        if set(restore_plans) != expected or any(
            not isinstance(value, bytes) or not value
            for value in restore_plans.values()
        ):
            raise ValueError(
                f"restore plans for {group!r} must cover logical partitions"
            )
        bound[group] = MappingProxyType(dict(restore_plans))
    return MappingProxyType(bound)


def bind_sparse_embedding_worker_restore_plans(
    *,
    partition_mappings: Mapping[str, SparsePartitionMapping],
    restore_plans_by_partition: Mapping[str, Mapping[int, bytes]],
) -> Mapping[str, LogicalPartitionBinding]:
    bound = bind_sparse_embedding_restore_plans(
        partition_mappings=partition_mappings,
        restore_plans_by_partition=restore_plans_by_partition,
    )
    return MappingProxyType(
        {
            group: LogicalPartitionBinding(
                restore_plans, axes=partition_mappings[group].sparse_axes
            )
            for group, restore_plans in bound.items()
        }
    )


def bind_sparse_checkpoint_restore_options(
    *,
    partition_mappings: Mapping[str, SparsePartitionMapping],
    worker_counts: Mapping[str, int],
    assignment_spec: PartitionSpec,
    key_dtype: str,
) -> Mapping[str, Mapping[int, Mapping[str, bytes]]]:
    if set(partition_mappings) != set(worker_counts):
        raise ValueError(
            "partition mappings and sparse worker counts must cover the same groups"
        )
    restore_plans_by_group = build_sparse_restore_plans(
        partition_mappings=partition_mappings,
        assignment_spec=assignment_spec,
        key_dtype=key_dtype,
    )
    bound: dict[str, Mapping[int, Mapping[str, bytes]]] = {}
    for group, mapping in partition_mappings.items():
        worker_count = worker_counts[group]
        if mapping.replica_axes:
            raise ValueError(
                f"sparse checkpoint restore requires one logical partition per worker: {group!r}"
            )
        if mapping.partition_count != worker_count:
            raise ValueError(
                f"sparse checkpoint restore mapping for {group!r} has "
                f"{mapping.partition_count} partitions for {worker_count} workers"
            )
        restore_plans = restore_plans_by_group[group]
        bound[f"Embedding_{group}"] = MappingProxyType(
            {
                rank: MappingProxyType({f"{group}_embedding": restore_plans[rank]})
                for rank in range(worker_count)
            }
        )
    return MappingProxyType(bound)


def _training_handles(
    node: NodeProxy,
    *,
    ranking_kwargs: Mapping[str, Any],
    sparse_groups: tuple[str, ...],
    side_info_groups: tuple[str, ...],
) -> dict[str, Any]:
    materialized = node.prepare_sparse_materialize(**dict(ranking_kwargs))
    bind_kwargs: dict[str, Any] = {
        "labels": materialized.labels,
        "samples_valid_count": materialized.samples_valid_count,
    }
    for group in sparse_groups:
        bind_kwargs[f"sparse_input_{group}"] = getattr(
            materialized, f"sparse_input_{group}"
        )
        bind_kwargs[f"padded_inverse_{group}"] = getattr(
            materialized, f"padded_inverse_{group}"
        )
    for group in side_info_groups:
        bind_kwargs[group] = getattr(materialized, group)
    jax_bound = node.prepare_jax_bind(**bind_kwargs, local=True)
    core_kwargs = {
        "jax_labels": jax_bound.jax_labels,
        "jax_samples_valid_count": jax_bound.jax_samples_valid_count,
    }
    for group in (*sparse_groups, *side_info_groups):
        core_kwargs[f"jax_input_{group}"] = getattr(jax_bound, f"jax_input_{group}")
    core = node.train_core(**core_kwargs, local=True)
    metrics = node.materialize_train_metrics(
        scores=core.scores,
        loss=core.loss,
        optimizer_update_window=core.optimizer_update_window,
        local=True,
    )
    scatter_kwargs: dict[str, Any] = {}
    for group in sparse_groups:
        scatter_kwargs[f"grad_jax_{group}"] = getattr(core, f"grad_jax_{group}")
        scatter_kwargs[f"emb_ref_{group}"] = getattr(materialized, f"emb_ref_{group}")
        scatter_kwargs[f"padded_inverse_{group}"] = getattr(
            materialized, f"padded_inverse_{group}"
        )
    scattered = node.scatter_sparse_grads(**scatter_kwargs, local=True)
    node.update(optimizer_update_window=core.optimizer_update_window, local=True)
    return {
        "node": node,
        "materialized": materialized,
        "jax_bound": jax_bound,
        "core": core,
        "metrics": metrics,
        "scattered": scattered,
    }


def _update_sparse_tables(
    *,
    tables: Mapping[str, Any],
    sparse_groups: tuple[str, ...],
    scattered: Mapping[str, Any],
    optimizer_update_window: Any,
) -> None:
    updates: dict[str, dict[str, Any]] = {}
    for group in sparse_groups:
        base_name, dimension = parse_sparse_group_name(group)
        updates.setdefault(base_name, {})[f"grads_dim{dimension}"] = getattr(
            scattered, f"grad_{group}"
        )
    for base_name, gradients in updates.items():
        if base_name not in tables:
            raise KeyError(f"missing sparse table for group {base_name!r}")
        tables[base_name].update(
            **gradients, optimizer_update_window=optimizer_update_window
        )


def _validate_static_groups(config: RankingTrainingAssemblyConfig) -> None:
    static = config.training_static_args
    if tuple(static["sparse_groups"]) != config.sparse_groups:
        raise ValueError(
            "training_static_args.sparse_groups must equal config.sparse_groups"
        )
    if tuple(static["side_info_groups"]) != config.side_info_groups:
        raise ValueError(
            "training_static_args.side_info_groups must equal config.side_info_groups"
        )


def add_ranking_training(
    graph: FunctionalGraph,
    *,
    reader: Any,
    input_node: Any,
    tables: Mapping[str, Any],
    model_arguments: Mapping[str, Any],
    processed: Any,
    config: RankingTrainingAssemblyConfig,
) -> RankingTrainingHandles:
    if not isinstance(graph, FunctionalGraph):
        raise TypeError("graph must be a Fluxway FunctionalGraph")
    if not isinstance(config.compute_spmd_name, str) or not config.compute_spmd_name:
        raise ValueError("compute_spmd_name must be a non-empty string")
    if not isinstance(config.metrics_spmd_name, str) or not config.metrics_spmd_name:
        raise ValueError("metrics_spmd_name must be a non-empty string")
    if reader is None or input_node is None:
        raise TypeError("reader and input_node must be Recipe-owned native handles")
    if not isinstance(tables, Mapping):
        raise TypeError("tables must be a mapping of Recipe-owned native handles")
    if not isinstance(model_arguments, Mapping):
        raise TypeError("model_arguments must be a mapping")
    _validate_static_groups(config)
    training_node = graph.node(
        config.compute_spmd_name,
        config.training_spmd,
        mesh=config.training_mesh,
        **dict(config.training_static_args),
    )
    training = _training_handles(
        training_node,
        ranking_kwargs=model_arguments,
        sparse_groups=config.sparse_groups,
        side_info_groups=config.side_info_groups,
    )
    metrics: Any | None = None
    if config.metrics_spmd is None:
        if config.metrics_mesh is not None or config.metrics_static_args is not None:
            raise ValueError(
                "metrics_mesh and metrics_static_args must be None when metrics_spmd is None"
            )
    else:
        if config.metrics_mesh is None or config.metrics_static_args is None:
            raise ValueError(
                "metrics_mesh and metrics_static_args are required when metrics_spmd is set"
            )
        metrics_node = graph.node(
            config.metrics_spmd_name,
            config.metrics_spmd,
            mesh=config.metrics_mesh,
            **dict(config.metrics_static_args),
        )
        metrics = metrics_node.step(
            scores=training["metrics"].scores,
            loss=training["metrics"].loss,
            metric_sampled=training["metrics"].metric_sampled,
            labels=processed.labels,
            group_ids=processed.group_id,
            samples_valid_count=processed.samples_valid_count,
        )
    _update_sparse_tables(
        tables=tables,
        sparse_groups=config.sparse_groups,
        scattered=training["scattered"],
        optimizer_update_window=training["core"].optimizer_update_window,
    )
    return RankingTrainingHandles(
        training=training, metrics=metrics, training_node=training_node
    )


def assemble_ranking_training_graph(
    graph: FunctionalGraph,
    *,
    graph_inputs: Any,
    training_spmd: type,
    training_mesh: DeviceMesh,
    training_static_args: Mapping[str, Any],
    metrics_spmd: type | None = None,
    metrics_mesh: DeviceMesh | None = None,
    metrics_static_args: Mapping[str, Any] | None = None,
    placement_recipe_id: str,
    spmd_placement_policy: Literal["balanced", "feasible"] = "balanced",
    compute_spmd_name: str,
    metrics_spmd_name: str,
    excluded_affinity_tables: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    handles = add_ranking_training(
        graph,
        reader=graph_inputs.reader,
        input_node=graph_inputs.input_node,
        tables=graph_inputs.tables,
        model_arguments=graph_inputs.model_arguments,
        processed=graph_inputs.processed,
        config=RankingTrainingAssemblyConfig(
            sparse_groups=tuple(training_static_args["sparse_groups"]),
            side_info_groups=tuple(training_static_args["side_info_groups"]),
            training_spmd=training_spmd,
            training_mesh=training_mesh,
            training_static_args=training_static_args,
            metrics_spmd=metrics_spmd,
            metrics_mesh=metrics_mesh,
            metrics_static_args=metrics_static_args,
            compute_spmd_name=compute_spmd_name,
            metrics_spmd_name=metrics_spmd_name,
        ),
    )
    task_graph = graph.build()
    placement = build_affinity_placement(
        recipe_id=placement_recipe_id,
        compiled_graph=task_graph.require_compiled_graph(),
        reader_spmd=graph_inputs.reader,
        input_spmd=graph_inputs.input_node,
        sparse_table_spmds=graph_inputs.tables,
        compute_spmd=handles.training_node,
        policy=spmd_placement_policy,
        excluded_affinity_tables=excluded_affinity_tables,
    )
    return {
        "training": handles.training,
        "metrics": handles.metrics,
        "placement": placement,
        "task_graph": task_graph,
    }


def assemble_ranking_evaluation_graph(
    graph: FunctionalGraph,
    *,
    graph_inputs: Any,
    evaluation_spmd: type,
    evaluation_mesh: DeviceMesh,
    evaluation_static_args: Mapping[str, Any],
    metrics_spmd: type,
    metrics_mesh: DeviceMesh,
    metrics_static_args: Mapping[str, Any],
    placement_recipe_id: str,
    compute_spmd_name: str,
    metrics_spmd_name: str,
    spmd_placement_policy: Literal["balanced", "feasible"] = "balanced",
    excluded_affinity_tables: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    evaluation_node = graph.node(
        compute_spmd_name,
        evaluation_spmd,
        mesh=evaluation_mesh,
        **dict(evaluation_static_args),
    )
    evaluation = evaluation_node.step(**dict(graph_inputs.model_arguments))
    metrics_node = graph.node(
        metrics_spmd_name,
        metrics_spmd,
        mesh=metrics_mesh,
        **dict(metrics_static_args),
    )
    metrics = metrics_node.step(
        scores=evaluation.scores,
        labels=evaluation.labels,
        group_ids=graph_inputs.processed.group_id,
        samples_valid_count=graph_inputs.processed.samples_valid_count,
    )
    task_graph = graph.build()
    placement = build_affinity_placement(
        recipe_id=placement_recipe_id,
        compiled_graph=task_graph.require_compiled_graph(),
        reader_spmd=graph_inputs.reader,
        input_spmd=graph_inputs.input_node,
        sparse_table_spmds=graph_inputs.tables,
        compute_spmd=evaluation_node,
        policy=spmd_placement_policy,
        excluded_affinity_tables=excluded_affinity_tables,
    )
    return {
        "evaluation": evaluation,
        "metrics": metrics,
        "placement": placement,
        "task_graph": task_graph,
    }


__all__ = [
    "RankingTrainingAssemblyConfig",
    "RankingTrainingHandles",
    "add_ranking_training",
    "assemble_ranking_evaluation_graph",
    "assemble_ranking_training_graph",
    "bind_sparse_checkpoint_restore_options",
    "bind_sparse_embedding_restore_plans",
    "bind_sparse_embedding_worker_restore_plans",
    "bind_sparse_worker_meshes",
]
