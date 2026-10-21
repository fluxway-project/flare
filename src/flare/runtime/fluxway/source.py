"""Fluxway-owned source descriptors and assigned source readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from flare.data.pipeline.consumption.source.batching.policies import (
    CompletionPolicy as FlareCompletionPolicy,
)
from flare.data.pipeline.consumption.source.contracts.model import (
    SourceRecordRange,
    normalize_locator,
)
from flare.data.pipeline.consumption.source.contracts.source import (
    ControlledSourceScanAdapter,
    SourceFactory,
    SourceReader,
)
from flare.data.pipeline.consumption.source.execution.invocation import invoke_source
from flare.data.pipeline.consumption.source.execution.materialization import (
    materialize_source_ranges,
    validate_source_read_concurrency,
)
from flare.data.schema.virtual_pad import (
    ArrowVirtualPadBatchFactory,
    VirtualPadBatchFactory,
)
from fluxway import profiler
from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceBindingIdentity,
    SourceWorkItem,
    source_work_global_valid_count,
)
from fluxway.common.metadata import unpack_metadata
from fluxway.common.types.configs import SPMDConfig
from fluxway.control_plane.components import DriverComponentContext
from fluxway.control_plane.ingestion import (
    AssignmentPolicy,
    IngestionLayoutActivationComponent,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
    ingestion_worker_id,
)
from fluxway.control_plane.ingestion.adapter import SourceScanBinding

@dataclass(frozen=True, slots=True)
class WorkItemScheduledDatasetSourceDescriptor:
    """Describe a finite dataset source planned by Fluxway."""

    requires_source_control_runtime = True

    source_factory: SourceFactory
    global_batch_size: int
    completion_policy: FlareCompletionPolicy | str
    source_assignment_policy: AssignmentPolicy = field(
        default_factory=AssignmentPolicy.default
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_factory, SourceFactory):
            raise TypeError("source descriptor requires SourceFactory")
        if type(self.global_batch_size) is not int or self.global_batch_size <= 0:
            raise ValueError("dataset global_batch_size must be positive")
        FlareCompletionPolicy.from_value(self.completion_policy)
        object.__setattr__(
            self,
            "source_assignment_policy",
            AssignmentPolicy.from_value(self.source_assignment_policy),
        )

    @staticmethod
    def _tail_layout_mode(policy: str | FlareCompletionPolicy) -> TailLayoutMode:
        resolved = FlareCompletionPolicy.from_value(policy)
        return {
            FlareCompletionPolicy.DROP: TailLayoutMode.DISCARD_REMAINDER,
            FlareCompletionPolicy.CARRY: TailLayoutMode.RETAIN_PENDING,
            FlareCompletionPolicy.REQUIRE: TailLayoutMode.REJECT_REMAINDER,
            FlareCompletionPolicy.PAD: TailLayoutMode.SYNTHESIZE_PADDING,
        }[resolved]

    def _build_reader_topology(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
        reader_node_ids: tuple[str, ...] | None = None,
    ) -> ReaderTopologyDescriptor:
        cycle = (
            config.require_graph_view().operator(operator_name).execution_layout.cycle
        )
        active_count = len(cycle[0])
        if reader_node_ids is not None and len(reader_node_ids) != config.num_workers:
            raise ValueError(
                "reader_node_ids must contain one attested node id per reader worker"
            )
        if any(len(group) != active_count for group in cycle):
            raise ValueError("dataset source requires uniform active reader counts")
        if self.global_batch_size % active_count:
            raise ValueError("global_batch_size must divide active reader count")
        return ReaderTopologyDescriptor(
            global_cardinality=self.global_batch_size,
            groups=tuple(
                ReaderGroupDemand(
                    ingestion_worker_id(spmd_id, operator_name, rank),
                    (
                        RankDemand(
                            ingestion_worker_id(spmd_id, operator_name, rank),
                            self.global_batch_size // active_count,
                        ),
                    ),
                    None if reader_node_ids is None else reader_node_ids[rank],
                )
                for rank in range(config.num_workers)
            ),
            execution_cycle=cycle,
        )

    def _build_source_scan_adapter(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> ControlledSourceScanAdapter:
        del spmd_id, operator_name
        source_options = config.static_args["source_options"]
        if not isinstance(source_options, Mapping):
            raise TypeError("controlled source graph requires a source_options mapping")
        adapter = self.source_factory.create_scan_planner(source_options)
        if not isinstance(adapter, ControlledSourceScanAdapter):
            raise TypeError(
                "source factory scan planner must implement ControlledSourceScanAdapter"
            )
        return adapter

    def build_source_scan_binding(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> SourceScanBinding:
        adapter = self._build_source_scan_adapter(
            spmd_id=spmd_id, operator_name=operator_name, config=config
        )
        binding_identity = SourceBindingIdentity.from_checkpoint(
            adapter.binding_identity
        )

        def initial_cursor_factory() -> OpaqueSourceCursor | None:
            return adapter.initial_scan_cursor()

        return SourceScanBinding(
            adapter=adapter,
            identity=binding_identity,
            initial_cursor_factory=initial_cursor_factory,
            topology=self._build_reader_topology(
                spmd_id=spmd_id, operator_name=operator_name, config=config
            ),
            tail_layout_mode=self._build_source_scan_tail_layout(
                spmd_id=spmd_id, operator_name=operator_name, config=config
            ),
            assignment_policy=self.source_assignment_policy,
        )

    def _build_source_scan_tail_layout(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> TailLayoutMode:
        del spmd_id, operator_name, config
        return self._tail_layout_mode(self.completion_policy)

    def build_component(
        self, context: DriverComponentContext
    ) -> IngestionLayoutActivationComponent:
        return IngestionLayoutActivationComponent(context)

    def create_session(
        self,
        options: Mapping[str, Any],
        *,
        pad_batch_factory: VirtualPadBatchFactory = ArrowVirtualPadBatchFactory.create_batch,
    ) -> AssignedSourceBatchReader:
        return AssignedSourceBatchReader(
            reader=self.source_factory.create_reader(options),
            projection=options["projection"],
            max_concurrent_reads=options["source_read_concurrency"],
            pad_batch_factory=pad_batch_factory,
        )


class AssignedSourceBatchReader:
    """Materialize exactly the SourceWorkItems acquired by Fluxway."""

    def __init__(
        self,
        *,
        reader: SourceReader,
        projection: Sequence[str],
        max_concurrent_reads: int = 2,
        pad_batch_factory: VirtualPadBatchFactory = ArrowVirtualPadBatchFactory.create_batch,
    ) -> None:
        if not isinstance(reader, SourceReader):
            raise TypeError("assigned source reader requires SourceReader")
        self._reader = reader
        self._projection = tuple(projection)
        self._concurrency = validate_source_read_concurrency(max_concurrent_reads)
        if not callable(pad_batch_factory):
            raise TypeError("assigned source reader requires a pad batch factory")
        self._pad_batch_factory = pad_batch_factory
        self._closed = False
        self._bound_source_identity: str | None = None
        self._cached_schema: pa.Schema | None = None

    @property
    def schema(self) -> pa.Schema:
        if self._cached_schema is not None:
            return self._cached_schema
        reader_schema = getattr(self._reader, "schema", None)
        if not isinstance(reader_schema, pa.Schema):
            raise ValueError(
                "padding-only source reader requires an explicit Arrow schema"
            )
        self._cached_schema = reader_schema
        return reader_schema

    @property
    def bound_source_identity(self) -> str | None:
        return self._bound_source_identity

    def bind_source_identity(self, source_identity: str) -> None:
        if (
            self._bound_source_identity is not None
            and self._bound_source_identity != source_identity
        ):
            raise ValueError("assigned source identity changed during reader session")
        if self._bound_source_identity is None:
            self._reader.bind_source_identity(source_identity)
            self._bound_source_identity = source_identity

    async def read(
        self,
        step: int,
        *,
        work_items: Sequence[SourceWorkItem],
    ) -> pa.Table | Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("assigned source reader is closed")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if not work_items:
            raise ValueError("assigned source reader requires SourceWorkItem values")

        real_items: list[SourceWorkItem] = []
        pad_items: list[SourceWorkItem] = []
        for item in work_items:
            if not isinstance(item, SourceWorkItem):
                raise TypeError(f"expected SourceWorkItem, got {type(item).__name__}")
            if item.key.step != step:
                raise ValueError(
                    f"source work item step {item.key.step} does not match current step {step}"
                )
            self.bind_source_identity(item.assignment.source_identity)
            if item.is_synthetic_padding:
                pad_items.append(item)
            else:
                real_items.append(item)

        total_slots = sum(item.cardinality for item in work_items)
        global_valid_count = source_work_global_valid_count(work_items)
        pad_rows = sum(item.cardinality for item in pad_items)
        table: pa.Table | None = None
        if real_items:
            with profiler.scope("sourcework.validate_assignments"):
                ranges: list[SourceRecordRange] = []
                for item in real_items:
                    assignment = item.assignment
                    opaque_token = assignment.descriptor.opaque_token
                    if not isinstance(opaque_token, bytes):
                        raise TypeError("source assignment opaque_token must be bytes")
                    record_range = SourceRecordRange.from_mapping(
                        unpack_metadata(opaque_token)
                    )
                    if (
                        record_range.source_identity != assignment.source_identity
                        or record_range.schema_digest != assignment.schema_digest
                        or record_range.revision_identity
                        != assignment.descriptor.revision_identity
                        or normalize_locator(record_range.locator).canonical()
                        != assignment.descriptor.physical_identity
                    ):
                        raise ValueError(
                            "source range metadata conflicts with assignment"
                        )
                    ranges.append(
                        record_range.slice(
                            item.logical_start - record_range.record_start,
                            item.cardinality,
                        )
                    )

            with profiler.scope("sourcework.materialize_ranges"):
                table = await materialize_source_ranges(
                    ranges,
                    read=self._reader.read,
                    projection=self._projection,
                    max_concurrent_reads=self._concurrency,
                )
            self._cached_schema = table.schema
            if pad_rows > 0:
                pad_table = pa.Table.from_batches(
                    [self._pad_batch_factory(table.schema, self._projection, pad_rows)]
                )
                table = pa.concat_tables([table, pad_table])
        else:
            schema = await invoke_source(lambda: self.schema)
            table = pa.Table.from_batches(
                [self._pad_batch_factory(schema, self._projection, pad_rows)]
            )

        if not isinstance(table, pa.Table) or table.num_rows != total_slots:
            raise ValueError("materialized rows do not match acquired source work")

        existing_meta = dict(table.schema.metadata or {})
        existing_meta[b"samples_valid_count"] = str(global_valid_count).encode("utf-8")
        return table.replace_schema_metadata(existing_meta)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await invoke_source(self._reader.close)


__all__ = [
    "AssignedSourceBatchReader",
    "WorkItemScheduledDatasetSourceDescriptor",
]
