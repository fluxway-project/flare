from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fluxway.common.types.checkpoint import CheckpointRetentionPolicy
from fluxway.control_plane.ingestion import AssignmentPolicy
from sparse_embedding.checkpoint import PartitionSpec

from flare.data.contracts.lineage import LineageConfig
from flare.data.pipeline.consumption.source.contracts.source import SourceFactory
from flare.data.schema.feature_catalog import FeatureCatalog, FeatureOutputKind
from flare.data.schema.feature_layout import build_training_feature_layout
from flare.mesh import bind_sparse_worker_meshes, build_1d_device_mesh
from flare.runtime.fluxway.resources.resolution import (
    ResourceProfiles,
    resolve_resource_profiles,
)
from flare.runtime.fluxway.source import (
    WorkItemScheduledDatasetSourceDescriptor,
)
from flare.runtime.fluxway.workers.dynamic_embedding import (
    get_sparse_embedding_class as get_sparse_table_class,
)
from flare.runtime.fluxway.workers.reader_contracts import (
    build_reader_source,
)
from flare.runtime.fluxway.workers.sparse_input import get_input_spmd_class
from flare.runtime.sparse_precision import require_key_dtype
from recipes.recsys.ranking.schema.sparse_features import (
    extract_sparse_groups_and_dimensions,
    parse_sparse_workers,
)
from recipes.recsys.ranking.task import RankingTask
from recipes.recsys.ranking.topology import bind_sparse_embedding_worker_restore_plans
from recipes.recsys.ranking.workers.payloads import (
    DynamicDataTypes,
    build_ranking_fluxway_data_types,
)
from recipes.recsys.ranking.workers.reader import (
    RankingReaderConfig,
    get_recsys_reader_class,
    resolve_reader_async_policy,
)


@dataclass(frozen=True)
class PipelineNodes:
    reader_spmd: type
    input_spmd: type
    sparse_table_factory: Callable[[str, list[int], int], type]
    sparse_groups: tuple[str, ...]
    dimensions_by_base: Mapping[str, list[int]]
    data_types: DynamicDataTypes
    resources: Any
    source_factory: SourceFactory
    source_descriptor: WorkItemScheduledDatasetSourceDescriptor | None
    reader_execution_groups: tuple[tuple[int, ...], ...] | None
    optimizer_enabled: bool
    sparse_partition_assignment_spec: PartitionSpec
    key_dtype: str


@dataclass(frozen=True)
class PipelineInputs:
    tables: Mapping[str, Any]
    model_arguments: Mapping[str, Any]
    processed: Any
    reader: Any
    input_node: Any


def build_pipeline_nodes(
    mb: Any,
    args: Any,
    *,
    source_factory: SourceFactory,
    manifest_source_enabled: bool,
    source_lineage: LineageConfig | None = None,
    optimizer_enabled: bool,
    load_optimizer_state: bool,
    retention_policy: CheckpointRetentionPolicy,
    runtime_env: dict | None = None,
    resource_profiles: ResourceProfiles | None = None,
    sparse_partition_assignment_spec: PartitionSpec,
    key_dtype: str,
) -> PipelineNodes:
    if not isinstance(optimizer_enabled, bool):
        raise TypeError("optimizer_enabled must be bool")
    if not isinstance(load_optimizer_state, bool):
        raise TypeError("load_optimizer_state must be bool")
    if not isinstance(sparse_partition_assignment_spec, PartitionSpec):
        raise TypeError("sparse_partition_assignment_spec must be a PartitionSpec")
    key_dtype = require_key_dtype(key_dtype)
    if not isinstance(retention_policy, CheckpointRetentionPolicy):
        raise TypeError("retention_policy must be CheckpointRetentionPolicy")
    if resource_profiles is not None and not isinstance(
        resource_profiles, ResourceProfiles
    ):
        raise TypeError("resource_profiles must be ResourceProfiles")
    sparse_groups, dimensions_by_base = extract_sparse_groups_and_dimensions(
        mb.pack_meta.sparse_id_groups
    )
    data_types = build_ranking_fluxway_data_types(sparse_groups)
    sparse_workers_config = parse_sparse_workers(args.sparse_workers, sparse_groups)
    if resource_profiles is not None:
        resources = resource_profiles
    else:
        worker_counts = {
            "driver": 1,
            "reader": args.reader_workers,
            "input": args.dp,
            "compute": args.dp,
            "table": sum(sparse_workers_config.values()),
            "metrics": 1,
        }
        resources = resolve_resource_profiles(
            worker_counts,
            compute_device="cpu" if args.jax_platform == "cpu" else "gpu",
        )

    local_pipeline_depth = int(args.sparse_max_staleness_steps) + 1
    if local_pipeline_depth <= 0:
        raise ValueError(
            "sparse_max_staleness_steps must produce a positive local pipeline "
            f"depth, got {args.sparse_max_staleness_steps}"
        )

    source_descriptor = None
    execution_groups = None
    if manifest_source_enabled:
        reader_count = args.reader_workers
        input_count = args.dp
        if reader_count % input_count:
            raise ValueError(
                "reader_workers must divide evenly by dp for whole-unit ingestion"
            )
        depth = reader_count // input_count
        execution_groups = tuple(
            tuple(lane * depth + slot for lane in range(input_count))
            for slot in range(depth)
        )
        source_assignment_policy_map = {
            "none": AssignmentPolicy(
                affinity_contiguous_chunks=False, affinity_same_file=False
            ),
            "contiguous_only": AssignmentPolicy(
                affinity_contiguous_chunks=True, affinity_same_file=False
            ),
            "same_file_only": AssignmentPolicy(
                affinity_contiguous_chunks=False, affinity_same_file=True
            ),
            "both": AssignmentPolicy(
                affinity_contiguous_chunks=True, affinity_same_file=True
            ),
        }
        if args.source_assignment_policy not in source_assignment_policy_map:
            raise ValueError(
                f"unsupported source_assignment_policy: {args.source_assignment_policy!r}"
            )
        source_descriptor = WorkItemScheduledDatasetSourceDescriptor(
            source_factory=source_factory,
            lineage=source_lineage,
            global_batch_size=args.global_batch_size,
            completion_policy=args.completion_policy,
            source_assignment_policy=source_assignment_policy_map[
                args.source_assignment_policy
            ],
        )
    reader_spmd = get_recsys_reader_class(
        sparse_groups=sparse_groups,
        runtime_env=runtime_env,
        resource_profile=resources.get_role("reader"),
        reader_workers=args.reader_workers,
        async_policy=resolve_reader_async_policy(local_pipeline_depth),
        source_descriptor=source_descriptor,
        interleave_unit_size=args.dp if manifest_source_enabled else 1,
        execution_groups=execution_groups,
    )
    input_spmd = get_input_spmd_class(
        sparse_groups=sparse_groups,
        runtime_env=runtime_env,
        resource_profile=resources.get_role("input"),
        batch_shardings={
            "side_info": {"type": "static", "axis": ["dp"]},
            "labels": {"type": "static", "axis": ["dp"]},
            "group_id": {"type": "static", "axis": ["dp"]},
            "samples_valid_count": {"type": "static", "axis": []},
        },
        key_dtype=key_dtype,
        sparse_partition_assignment_spec=sparse_partition_assignment_spec,
        pipeline_depth=local_pipeline_depth,
    )

    def sparse_table_factory(name: str, dims: list[int], shards: int) -> type:
        return get_sparse_table_class(
            name,
            dims,
            shards,
            resources.get_role("table"),
            runtime_env,
            args.sparse_ckpt_backend,
            args.sparse_max_staleness_steps,
            optimizer_enabled,
            load_optimizer_state=optimizer_enabled and load_optimizer_state,
            sparse_partition_assignment_spec=sparse_partition_assignment_spec,
            value_dtype=args.precision.sparse_storage_dtype_name,
            gradient_dtype=args.precision.sparse_gradient_dtype_name,
            gradient_transport_dtype=args.precision.sparse_gradient_transport_dtype_name,
            optimizer_dtype=args.precision.sparse_optimizer_dtype_name,
            key_dtype=key_dtype,
        )

    return PipelineNodes(
        reader_spmd=reader_spmd,
        input_spmd=input_spmd,
        sparse_table_factory=sparse_table_factory,
        sparse_groups=sparse_groups,
        dimensions_by_base=dimensions_by_base,
        data_types=data_types,
        resources=resources,
        source_factory=source_factory,
        source_descriptor=source_descriptor,
        reader_execution_groups=execution_groups,
        optimizer_enabled=optimizer_enabled,
        sparse_partition_assignment_spec=sparse_partition_assignment_spec,
        key_dtype=key_dtype,
    )


def build_pipeline_inputs(
    *,
    mb: Any,
    side_info_seq: Any,
    args: Any,
    catalog_token: str | None,
    common: PipelineNodes,
    sparse_workers_config: Mapping[str, int],
    repeat_first_batch: bool,
    execution_tracer: Any = None,
    shuffle: bool = True,
    num_epochs: int = 10,
    cached_batch_metric_contract: Mapping[str, Any] | None = None,
    ranking_task: RankingTask,
    pretrained_weight_source: Any | None = None,
) -> PipelineInputs:
    if not isinstance(ranking_task, RankingTask):
        raise TypeError("ranking_task must be a RankingTask")
    sparse_restore_payload = None
    pretrained_sparse_meshes = None
    if pretrained_weight_source is not None:
        restore_plans_by_partition = getattr(
            pretrained_weight_source, "restore_payloads_by_partition", None
        )
        partition_mappings = getattr(
            pretrained_weight_source, "partition_mappings", None
        )
        if restore_plans_by_partition is None or partition_mappings is None:
            raise RuntimeError(
                "inference weight composition must provide runtime topology partition mapping and "
                "per-rank restore payloads"
            )
        sparse_restore_payload = bind_sparse_embedding_worker_restore_plans(
            partition_mappings=partition_mappings,
            restore_plans_by_partition=restore_plans_by_partition,
        )
        pretrained_sparse_meshes = bind_sparse_worker_meshes(
            partition_mappings=partition_mappings,
            worker_counts=sparse_workers_config,
        )
    reader_mesh = build_1d_device_mesh(args.reader_workers, axis_name="worker")
    input_mesh = build_1d_device_mesh(args.dp, axis_name="dp")

    projection_side_info = list(side_info_seq)
    projection_sparse = {
        group: list(features)
        for group, features in mb.pack_meta.sparse_id_groups.items()
    }
    feature_layout = build_training_feature_layout(
        FeatureCatalog.from_manifest_dir(args.feature_manifest_dir),
        projection_side_info,
        projection_sparse,
        [ranking_task.group_id_column],
        [ranking_task.label_column],
        system_columns={
            ranking_task.label_column: FeatureOutputKind.DENSE_I32_INDICES,
        },
        null_index_sentinel=args.null_index_sentinel,
    )
    batch_transformer = ranking_task.batch_transformer_factory(
        side_info_seq=side_info_seq,
        categorical_feature_groups=projection_sparse,
    )(feature_layout)
    if not callable(batch_transformer):
        raise TypeError("batch_transformer_factory must return a callable transformer")
    if common.source_descriptor is None:
        raise ValueError("ranking reader requires an assigned source descriptor")
    source = build_reader_source(
        descriptor=common.source_descriptor,
        projection=feature_layout.projection,
        source_read_concurrency=args.source_read_concurrency,
        storage_options={
            "dataset_path": args.dataset_path,
            "storage_format": args.storage_format,
            **(
                {"file_storage_options": args.file_storage_options}
                if args.storage_format in {"parquet", "vortex"}
                else {}
            ),
            "catalog_name": args.catalog_name,
            "catalog_type": args.catalog_type,
            "catalog_uri": args.catalog_uri,
            "warehouse": args.warehouse,
            "iceberg_config": args.iceberg_config,
            "iceberg_snapshot_id": args.iceberg_snapshot_id,
            "catalog_token": catalog_token,
            "table_identifier": args.table_identifier,
            "iceberg_max_open_files": args.iceberg_max_open_files,
        },
    )
    reader_config = RankingReaderConfig(
        source=source,
        feature_layout=feature_layout,
        sparse_groups=frozenset(common.sparse_groups),
        transform=batch_transformer,
        output_type=common.data_types.ranking_reader_output,
        group_id_column=ranking_task.group_id_column,
        repeat_first_batch=repeat_first_batch,
        execution_audit_path=args.reader_execution_audit_path,
    )
    reader = common.reader_spmd(
        name="Reader",
        mesh=reader_mesh,
        reader_config=reader_config,
        source_options=source.options,
    )
    input_node = common.input_spmd(
        name="Input",
        mesh=input_mesh,
        sparse_groups=common.sparse_groups,
        InputNodeOutput=common.data_types.input_node_output,
        key_dtype=common.key_dtype,
        sparse_partition_assignment_spec=common.sparse_partition_assignment_spec,
        execution_tracer=execution_tracer,
    )

    dimensions_by_base = common.dimensions_by_base

    tables = {}
    for base_name, dimensions in dimensions_by_base.items():
        shards = sparse_workers_config[base_name]
        sparse_mesh = (
            pretrained_sparse_meshes[base_name]
            if pretrained_sparse_meshes is not None
            else build_1d_device_mesh(shards, axis_name="shard")
        )
        table_type = common.sparse_table_factory(base_name, dimensions, shards)
        tables[base_name] = table_type(
            name=f"Embedding_{base_name}",
            mesh=sparse_mesh,
            gname=base_name,
            spec={
                "emb_dims": dimensions,
                "value_dtype": args.precision.sparse_storage_dtype_name,
                "gradient_dtype": args.precision.sparse_gradient_dtype_name,
                "gradient_transport_dtype": args.precision.sparse_gradient_transport_dtype_name,
                "optimizer_dtype": args.precision.sparse_optimizer_dtype_name,
                "grad_accum_steps": args.grad_accum_steps,
                "sparse_reserve_capacity": args.sparse_reserve_capacity,
                "weight_source": pretrained_weight_source,
                "sparse_restore_payload": (
                    sparse_restore_payload.get(base_name)
                    if sparse_restore_payload is not None
                    else None
                ),
                "key_dtype": common.key_dtype,
                "optimizer_enabled": common.optimizer_enabled,
                **(
                    {"sparse_optimizer": args.sparse_optimizer}
                    if common.optimizer_enabled
                    else {}
                ),
            },
            LookupOutput=common.data_types.lookup_outputs[base_name],
            validate_tensor_values=args.validate_tensor_values,
        )

    raw = reader.read()
    input_arguments = {name: getattr(raw, name) for name in common.sparse_groups}
    input_arguments.update(
        side_info=raw.side_info,
        labels=raw.labels,
        group_id=raw.group_id,
        samples_valid_count=raw.samples_valid_count,
    )
    processed = input_node.process(**input_arguments)

    model_arguments: dict[str, Any] = {}
    for base_name, dimensions in dimensions_by_base.items():
        lookup = tables[base_name].lookup(
            **{
                f"unique_ids_dim{dimension}": getattr(
                    processed, f"unique_{base_name}_dim_{dimension}"
                )
                for dimension in dimensions
            }
        )
        for dimension in dimensions:
            model_arguments[f"emb_{base_name}_dim_{dimension}"] = getattr(
                lookup, f"sparse_features_{base_name}_dim_{dimension}"
            )
            model_arguments[f"inv_{base_name}_dim_{dimension}"] = getattr(
                processed, f"inv_{base_name}_dim_{dimension}"
            )
    model_arguments["side_info"] = processed.side_info
    model_arguments["labels"] = processed.labels
    model_arguments["samples_valid_count"] = processed.samples_valid_count
    return PipelineInputs(
        tables=tables,
        model_arguments=model_arguments,
        processed=processed,
        reader=reader,
        input_node=input_node,
    )


__all__ = [
    "PipelineInputs",
    "PipelineNodes",
    "build_pipeline_inputs",
    "build_pipeline_nodes",
]
