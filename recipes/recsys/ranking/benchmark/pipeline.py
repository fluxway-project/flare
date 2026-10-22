from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fluxway.common.mesh import DeviceMesh
from fluxway.common.transfer import WholeUnitRouting

from flare.data.schema.feature_catalog import FeatureCatalog, FeatureOutputKind
from flare.data.schema.feature_layout import build_training_feature_layout
from flare.runtime.fluxway.workers.reader_contracts import (
    build_reader_source,
)
from recipes.recsys.ranking.workers.reader import (
    RankingReaderConfig,
)


@dataclass(frozen=True)
class ThroughputPipelineInputs:
    reader: Any
    input_node: Any
    sink_node: Any
    processed: Any


def build_throughput_pipeline_inputs(
    *,
    mb: Any,
    side_info_seq: Sequence[str],
    args: Any,
    common: Any,
    sink_spmd: type,
    repeat_first_batch: bool,
    execution_tracer: Any = None,
    shuffle: bool = False,
    num_epochs: int = 1,
    batch_transformer_factory: Callable[[Any], Callable[[Any], Mapping[str, Any]]],
    report_output_path: str | None = None,
    include_all_physical_columns: bool = True,
    metadata_cols: Sequence[str],
    label_column: str,
    group_id_column: str,
    extra_label_cols: Sequence[str],
    null_index_sentinel: int,
    execution_audit_path: str | None = None,
) -> ThroughputPipelineInputs:
    if not callable(batch_transformer_factory):
        raise TypeError("batch_transformer_factory must be callable")
    reader_mesh = DeviceMesh(
        device_shape=(args.reader_workers,), axis_names=("worker",)
    )
    input_mesh = DeviceMesh(device_shape=(args.dp,), axis_names=("dp",))
    sink_mesh = DeviceMesh(device_shape=(args.dp,), axis_names=("dp",))

    projection_side_info = list(side_info_seq)
    projection_sparse = {
        group: list(features)
        for group, features in mb.pack_meta.sparse_id_groups.items()
    }
    system_columns = {
        label_column: FeatureOutputKind.DENSE_I32_INDICES,
    }
    label_cols = [label_column]
    if include_all_physical_columns:
        for col in extra_label_cols:
            if col not in label_cols:
                label_cols.append(col)
                system_columns[col] = FeatureOutputKind.DENSE_I32_INDICES

    feature_layout = build_training_feature_layout(
        FeatureCatalog.from_manifest_dir(args.feature_manifest_dir),
        projection_side_info,
        projection_sparse,
        list(metadata_cols),
        label_cols,
        system_columns=system_columns,
        null_index_sentinel=null_index_sentinel,
    )
    batch_transformer = batch_transformer_factory(feature_layout)
    if not callable(batch_transformer):
        raise TypeError("batch_transformer_factory must return a callable transformer")
    if common.source_descriptor is None:
        raise ValueError("ranking reader requires an assigned source descriptor")
    storage_options: dict[str, Any] = {
        "dataset_path": args.dataset_path,
        "storage_format": args.storage_format,
        "catalog_name": args.catalog_name,
        "catalog_type": args.catalog_type,
        "catalog_uri": args.catalog_uri,
        "warehouse": args.warehouse,
        "iceberg_config": args.iceberg_config,
        "iceberg_snapshot_id": args.iceberg_snapshot_id,
        "table_identifier": args.table_identifier,
        "iceberg_max_open_files": args.iceberg_max_open_files,
    }
    if args.storage_format in {"parquet", "vortex"}:
        storage_options["file_storage_options"] = args.file_storage_options
    source = build_reader_source(
        descriptor=common.source_descriptor,
        projection=feature_layout.projection,
        source_read_concurrency=args.source_read_concurrency,
        storage_options=storage_options,
    )
    reader_config = RankingReaderConfig(
        source=source,
        feature_layout=feature_layout,
        sparse_groups=frozenset(common.sparse_groups),
        transform=batch_transformer,
        output_type=common.data_types.ranking_reader_output,
        group_id_column=group_id_column,
        repeat_first_batch=repeat_first_batch,
        execution_audit_path=execution_audit_path,
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
    sink_node = sink_spmd(
        name="ThroughputSink",
        mesh=sink_mesh,
        max_steps=args.steps,
        reader_workers=args.reader_workers,
        report_output_path=report_output_path,
    )

    raw = reader.read()
    input_arguments = {name: getattr(raw, name) for name in common.sparse_groups}
    input_arguments.update(
        side_info=raw.side_info,
        labels=raw.labels,
        group_id=raw.group_id,
        samples_valid_count=raw.samples_valid_count,
    )
    processed = input_node.process(
        **input_arguments, unit_routing=WholeUnitRouting(tuple(range(args.dp)))
    )
    sink_arguments = {
        "side_info": processed.side_info,
        "labels": processed.labels,
        "group_id": processed.group_id,
        "samples_valid_count": processed.samples_valid_count,
    }
    sink_node.consume(**sink_arguments)
    return ThroughputPipelineInputs(
        reader=reader,
        input_node=input_node,
        sink_node=sink_node,
        processed=processed,
    )


__all__ = [
    "ThroughputPipelineInputs",
    "build_throughput_pipeline_inputs",
]
