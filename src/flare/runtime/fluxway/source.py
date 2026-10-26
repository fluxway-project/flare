from __future__ import annotations

from builtins import BaseExceptionGroup
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from flare.data.contracts.lineage import LineageConfig
from flare.data.pipeline.consumption.source.batching.policies import (
    CompletionPolicy as FlareCompletionPolicy,
)
from flare.data.pipeline.consumption.source.contracts.source import (
    ControlledSourceScanAdapter,
    SourceFactory,
    SourceReadSession,
)
from flare.data.pipeline.consumption.source.execution.invocation import invoke_source
from flare.data.schema.virtual_pad import VirtualPadBatchFactory
from fluxway.common.ingestion import (
    SourceBindingIdentity,
)
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
    lineage: LineageConfig | None = field(default=None, kw_only=True)
    global_batch_size: int
    completion_policy: FlareCompletionPolicy | str
    source_assignment_policy: AssignmentPolicy = field(
        default_factory=AssignmentPolicy.default
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_factory, SourceFactory):
            raise TypeError("source descriptor requires SourceFactory")
        if self.lineage is not None and not isinstance(self.lineage, LineageConfig):
            raise TypeError("source lineage must be LineageConfig")
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

    async def _build_source_scan_adapter(
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
        adapter = await invoke_source(
            self.source_factory.create_scan_planner, source_options
        )
        if not isinstance(adapter, ControlledSourceScanAdapter):
            raise TypeError(
                "source factory scan planner must implement ControlledSourceScanAdapter"
            )
        return adapter

    async def build_source_scan_binding(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> SourceScanBinding:
        adapter = await self._build_source_scan_adapter(
            spmd_id=spmd_id, operator_name=operator_name, config=config
        )
        try:
            binding_identity = await invoke_source(
                lambda: SourceBindingIdentity.from_checkpoint(adapter.binding_identity)
            )
            return SourceScanBinding(
                adapter=adapter,
                identity=binding_identity,
                initial_cursor_factory=adapter.initial_scan_cursor,
                topology=self._build_reader_topology(
                    spmd_id=spmd_id, operator_name=operator_name, config=config
                ),
                tail_layout_mode=self._build_source_scan_tail_layout(
                    spmd_id=spmd_id, operator_name=operator_name, config=config
                ),
                assignment_policy=self.source_assignment_policy,
            )
        except BaseException as error:
            try:
                await invoke_source(adapter.close)
            except BaseException as cleanup_error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "source binding construction and cleanup failed",
                    [error, cleanup_error],
                )
            raise

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

    async def create_session(
        self,
        options: Mapping[str, Any],
        *,
        pad_batch_factory: VirtualPadBatchFactory | None = None,
    ) -> SourceReadSession:
        options = dict(options)
        if pad_batch_factory is not None:
            options["pad_batch_factory"] = pad_batch_factory
        session = await invoke_source(
            self.source_factory.create_reader,
            options,
            include_lineage=self.lineage is not None,
        )
        if not isinstance(session, SourceReadSession):
            raise TypeError("source factory must create SourceReadSession")
        return session


__all__ = ["WorkItemScheduledDatasetSourceDescriptor"]
