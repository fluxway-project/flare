from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from dataclasses import make_dataclass
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.ingestion import SourceWorkAssignment, SourceWorkItem
from fluxway.common.metadata import metadata_digest, pack_metadata
from fluxway.common.types import CheckpointCut
from fluxway.common.types.tensor import TransportTensor
from fluxway.control_plane.ingestion import (
    CompiledSegment,
    IngestionLayoutActivation,
    LayoutUnitDescriptor,
    WorkerLayoutDescriptor,
)
from fluxway.reader.ingestion_layout import SegmentReadChunk
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)

from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)
from flare.data.pipeline.consumption.source.contracts.source import SourceReader
from flare.runtime.fluxway.source import (
    AssignedSourceBatchReader,
    WorkItemScheduledDatasetSourceDescriptor,
)
from flare.runtime.fluxway.workers.reader import BatchReaderSPMD
from flare.runtime.fluxway.workers.reader_contracts import build_reader_source
from recipes.recsys.ranking.workers.reader import RecsysReaderSPMD


class DescriptorFactory:
    def create_scan_planner(self, options):
        raise AssertionError("planner must not be called")

    def create_reader(self, options):
        raise AssertionError("reader must not be called")


class LocalSegmentReader:
    def __init__(self) -> None:
        self.close_count = 0

    def read(self, request, *, batch_size):
        assert request.physical_identity == "local.parquet"
        yield SegmentReadChunk(
            {"value": np.asarray([3, 5], dtype=np.int32)},
            cardinality=2,
            physical_bytes=16,
            object_opened=True,
        )

    def close(self) -> None:
        self.close_count += 1


def _activation() -> IngestionLayoutActivation:
    descriptor = WorkerLayoutDescriptor(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        layout_digest="compiled-layout",
        worker_id="reader:read:worker:0",
        segments=(
            CompiledSegment(
                step=0,
                assignment_id="assignment-0",
                worker_id="reader:read:worker:0",
                unit_id="unit-0",
                revision_identity="revision",
                locality_key="node",
                physical_identity="local.parquet",
                cardinality_metric="rows",
                logical_start=0,
                logical_end=2,
                opaque_token="row-group-0",
            ),
        ),
    ).encode()
    return IngestionLayoutActivation(
        epoch=0,
        start_position=0,
        end_position=1,
        layout_digest="aggregate-layout",
        terminal=True,
        descriptor=descriptor,
    )


@pytest.mark.asyncio
async def test_direct_ingestion_reads_locally_and_returns_transport_tensor() -> None:
    read_session = LocalSegmentReader()
    reader = BatchReaderSPMD(
        source=build_reader_source(read_session=read_session, read_chunk_size=2),
        output_builder=dict,
    )
    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=None,
        global_source_offset=0,
    )

    await reader.activate_ingestion_layout(_activation())
    batch = await reader.read()

    assert isinstance(batch["value"], TransportTensor)
    assert np.array_equal(batch["value"].flat_values, np.asarray([3, 5]))
    assert batch["value"].origin_indices is not None
    exhausted = await reader.read()
    assert isinstance(exhausted, ExecutionResult)
    assert exhausted.status is ExecutionStatus.SOURCE_EXHAUSTED
    assert len(exhausted.control_requests) == 1
    assert exhausted.control_requests[0].consumed_cardinality == 2
    assert read_session.close_count == 1


@pytest.mark.asyncio
async def test_direct_ingestion_rejects_conflicting_activation() -> None:
    reader = BatchReaderSPMD(
        source=build_reader_source(
            read_session=LocalSegmentReader(), read_chunk_size=2
        ),
        output_builder=dict,
    )
    first = _activation()
    conflicting = IngestionLayoutActivation(
        epoch=first.epoch,
        start_position=first.start_position,
        end_position=first.end_position,
        layout_digest="other-layout",
        terminal=True,
        descriptor=first.descriptor,
    )

    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=None,
        global_source_offset=0,
    )
    await reader.activate_ingestion_layout(first)
    await reader.activate_ingestion_layout(first)
    with pytest.raises(ValueError, match="conflicting"):
        await reader.activate_ingestion_layout(conflicting)


def test_assigned_reader_requires_feature_layout() -> None:
    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=DescriptorFactory(),
        global_batch_size=2,
        completion_policy="drop",
    )

    with pytest.raises(ValueError, match="projection"):
        build_reader_source(
            descriptor=descriptor,
            source_read_concurrency=1,
            storage_options={},
        )


def test_assigned_reader_preserves_catalog_token_option() -> None:
    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=DescriptorFactory(),
        global_batch_size=2,
        completion_policy="drop",
    )
    source = build_reader_source(
        descriptor=descriptor,
        projection=("fid_1",),
        source_read_concurrency=1,
        storage_options={
            "storage_format": "iceberg",
            "catalog_name": "gray_metalake",
            "catalog_type": "rest",
            "catalog_uri": "http://catalog.example/iceberg",
            "table_identifier": "throughput.whw_input_graph",
            "catalog_token": "test-token",
        },
    )
    assert source.options["catalog_token"] == "test-token"


class DummySourceReader(SourceReader):
    def __init__(self) -> None:
        self.close_calls = 0
        self.read_calls: list[SourceRecordRange] = []
        self.bound_source_identity: str | None = None

    def bind_source_identity(self, source_identity: str) -> None:
        if (
            self.bound_source_identity is not None
            and self.bound_source_identity != source_identity
        ):
            raise ValueError("source identity changed")
        self.bound_source_identity = source_identity

    def read(
        self, record_range: SourceRecordRange, projection: Sequence[str]
    ) -> pa.Table:
        self.read_calls.append(record_range)
        return pa.table(
            {
                "value": list(
                    range(
                        record_range.record_start,
                        record_range.record_start + record_range.record_count,
                    )
                )
            }
        )

    def read_batches(
        self,
        record_range: SourceRecordRange,
        projection: Sequence[str],
        *,
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
        yield from self.read(record_range, projection).to_batches()

    def close(self) -> None:
        self.close_calls += 1


def _work_item(
    *,
    step: int,
    logical_start: int,
    logical_end: int,
    descriptor_start: int = 10,
    source_identity: str = "source",
) -> SourceWorkItem:
    record_range = SourceRecordRange(
        source_identity=source_identity,
        revision_identity="revision",
        schema_digest="schema",
        locator=FileLocator("file:///data/part.parquet"),
        record_start=10,
        record_count=20,
        record_format="parquet",
    )
    descriptor = LayoutUnitDescriptor(
        unit_id=metadata_digest(record_range.to_dict()),
        revision_identity=record_range.revision_identity,
        cardinality=record_range.record_count,
        cardinality_metric="rows",
        split_policy="SplittableSpan",
        locality_key=record_range.locator.canonical(),
        physical_identity=record_range.locator.canonical(),
        logical_start=record_range.record_start,
        opaque_token=pack_metadata(record_range.to_dict()),
    ).slice(
        descriptor_start - record_range.record_start,
        record_range.record_end - descriptor_start,
    )
    assignment = SourceWorkAssignment.create(
        epoch=0,
        step=step,
        source_identity=source_identity,
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        descriptor=descriptor,
        global_start=step * 20,
        global_batch_size=20,
        global_valid_count=20,
    )
    return SourceWorkItem(assignment, logical_start, logical_end)


@pytest.mark.asyncio
async def test_assigned_reader_materializes_exact_work_and_closes_once() -> None:
    source = DummySourceReader()
    reader = AssignedSourceBatchReader(
        reader=source,
        projection=("value",),
    )

    result = await reader.read(
        0,
        work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
    )

    assert isinstance(result, pa.Table)
    assert result["value"].to_pylist() == [12, 13, 14, 15, 16]
    assert source.read_calls[0].record_start == 12
    assert source.read_calls[0].record_count == 5
    await asyncio.gather(reader.close(), reader.close())
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_assigned_reader_binds_identity_once_and_rejects_changes() -> None:
    source = DummySourceReader()
    reader = AssignedSourceBatchReader(reader=source, projection=("value",))

    await reader.read(
        0,
        work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
    )
    assert source.bound_source_identity == "source"

    with pytest.raises(ValueError, match="identity changed"):
        await reader.read(
            1,
            work_items=(
                _work_item(
                    step=1,
                    logical_start=22,
                    logical_end=27,
                    descriptor_start=20,
                    source_identity="replacement",
                ),
            ),
        )


def test_reader_checkpoint_contains_only_the_controller_cut_position() -> None:
    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=DescriptorFactory(),
        global_batch_size=2,
        completion_policy="drop",
    )
    source = build_reader_source(
        descriptor=descriptor,
        projection=("value",),
        source_read_concurrency=1,
        storage_options={"storage_format": "iceberg"},
    )
    reader = BatchReaderSPMD(source=source, output_builder=dict)
    context = SimpleNamespace(cut=CheckpointCut.for_position("reader", 3))

    state = reader.capture_checkpoint_state(context)
    assert state == {"position": 3}

    restored = BatchReaderSPMD(source=source, output_builder=dict)
    restored.restore_checkpoint_state(state, context)
    assert restored._assigned_session is None


def test_direct_reader_checkpoint_contains_only_the_controller_cut_position() -> None:
    reader = BatchReaderSPMD(
        source=build_reader_source(
            read_session=LocalSegmentReader(), read_chunk_size=2
        ),
        output_builder=dict,
    )
    context = SimpleNamespace(cut=CheckpointCut.for_position("reader", 0))

    state = reader.capture_checkpoint_state(context)
    assert state == {"position": 0}
    reader.restore_checkpoint_state(state, context)


def test_reader_checkpoint_rejects_redundant_source_identity_state() -> None:
    reader = BatchReaderSPMD(
        source=build_reader_source(
            read_session=LocalSegmentReader(), read_chunk_size=2
        ),
        output_builder=dict,
    )
    context = SimpleNamespace(cut=CheckpointCut.for_position("reader", 0))

    with pytest.raises(ValueError, match="fields do not match"):
        reader.restore_checkpoint_state(
            {"position": 0, "source_identity": "source"}, context
        )


@pytest.mark.asyncio
async def test_restored_reader_rejects_replacement_assignment_before_materialize() -> (
    None
):
    class ReaderFactory:
        def create_scan_planner(self, options):
            raise AssertionError("planner must not be called")

        def create_reader(self, options):
            return DummySourceReader()

    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=ReaderFactory(),
        global_batch_size=20,
        completion_policy="drop",
    )
    source = build_reader_source(
        descriptor=descriptor,
        projection=("value",),
        source_read_concurrency=1,
        storage_options={"storage_format": "parquet"},
    )
    reader = BatchReaderSPMD(source=source, output_builder=dict)
    context = SimpleNamespace(cut=CheckpointCut.for_position("reader", 1))
    reader.restore_checkpoint_state({"position": 1}, context)
    await reader.activate_ingestion_layout(_activation())
    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        global_source_offset=0,
    )

    await reader.read()
    reader.ctx = SimpleNamespace(
        step=1,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=(
            _work_item(
                step=1,
                logical_start=22,
                logical_end=27,
                descriptor_start=20,
                source_identity="replacement",
            ),
        ),
        global_source_offset=0,
    )

    with pytest.raises(ValueError, match="identity changed"):
        await reader.read()


@pytest.mark.asyncio
async def test_assigned_reader_rejects_step_mismatch() -> None:
    reader = AssignedSourceBatchReader(
        reader=DummySourceReader(),
        projection=("value",),
    )

    with pytest.raises(ValueError, match="does not match current step"):
        await reader.read(
            1,
            work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        )


@pytest.mark.asyncio
async def test_assigned_reader_resolves_slice_from_original_file_range() -> None:
    source = DummySourceReader()
    reader = AssignedSourceBatchReader(reader=source, projection=("value",))

    result = await reader.read(
        0,
        work_items=(
            _work_item(
                step=0,
                logical_start=14,
                logical_end=19,
                descriptor_start=12,
            ),
        ),
    )

    assert result["value"].to_pylist() == [14, 15, 16, 17, 18]
    assert source.read_calls[0].record_start == 14
    assert source.read_calls[0].record_count == 5


@pytest.mark.asyncio
async def test_assigned_reader_preserves_file_rows_across_planning_windows() -> None:
    from dataclasses import replace

    from fluxway.control_plane.ingestion import (
        AlgebraicLayoutCompiler,
        AssignmentPolicy,
        LayoutCompileRequest,
        RankDemand,
        ReaderGroupDemand,
        ReaderTopologyDescriptor,
        TailLayoutMode,
    )
    from fluxway.control_plane.ingestion.planning_component import (
        _split_units_at_cardinality,
    )

    record_range = SourceRecordRange(
        source_identity="source",
        revision_identity="revision",
        schema_digest="schema",
        locator=FileLocator("file:///data/part.parquet"),
        record_start=10,
        record_count=8,
        record_format="parquet",
    )
    unit = LayoutUnitDescriptor(
        unit_id=metadata_digest(record_range.to_dict()),
        revision_identity=record_range.revision_identity,
        cardinality=record_range.record_count,
        cardinality_metric="rows",
        split_policy="SplittableSpan",
        locality_key=record_range.locator.canonical(),
        physical_identity=record_range.locator.canonical(),
        logical_start=record_range.record_start,
        opaque_token=pack_metadata(record_range.to_dict()),
    )
    request = LayoutCompileRequest(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(unit,),
        topology=ReaderTopologyDescriptor(
            4, (ReaderGroupDemand("reader", (RankDemand("rank", 4),)),)
        ),
        tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
        assignment_policy=AssignmentPolicy.default(),
    )
    windows = _split_units_at_cardinality((unit,), 4)
    source = DummySourceReader()
    reader = AssignedSourceBatchReader(reader=source, projection=("value",))
    planned = []
    actual = []
    try:
        for step, units in enumerate(windows):
            compiled = AlgebraicLayoutCompiler().compile(
                replace(request, units=units), epoch=step, start_step=step
            )
            work_items = tuple(
                SourceWorkItem(
                    assignment,
                    assignment.descriptor.logical_start,
                    assignment.descriptor.logical_start
                    + assignment.descriptor.cardinality,
                )
                for assignment in compiled.work_epoch.assignments
            )
            planned.extend(
                row
                for item in work_items
                for row in range(item.logical_start, item.logical_end)
            )
            batch = await reader.read(step, work_items=work_items)
            actual.extend(batch["value"].to_pylist())
        assert planned == list(range(10, 18))
        assert actual == list(range(10, 18))
    finally:
        await reader.close()


def _ranking_reader_for_batch(*, global_rows: int) -> RecsysReaderSPMD:
    reader = object.__new__(RecsysReaderSPMD)
    reader.group_id_column = "group_id"
    reader.sparse_groups = frozenset()
    reader._output_type = make_dataclass(
        "RankingReaderOutput",
        [("group_id", object), ("labels", object), ("samples_valid_count", object)],
    )
    reader._source_batch_coordinates = lambda rows: (
        ContinuousIndex.chunk(0, rows),
        global_rows,
    )
    return reader


def test_ranking_reader_uses_global_validity_count_for_partial_global_batch() -> None:
    reader = _ranking_reader_for_batch(global_rows=4)
    converted = reader._to_transport_batch(
        {
            "group_id": np.asarray([10, 11], dtype=np.int64),
            "labels": np.asarray([1, 0], dtype=np.int32),
            "samples_valid_count": np.asarray([3], dtype=np.int32),
        },
        logical_rank=0,
        logical_world=2,
    )

    assert converted["samples_valid_count"].flat_values.tolist() == [3]


def test_ranking_reader_requires_validity_metadata_without_scheduled_work() -> None:
    reader = _ranking_reader_for_batch(global_rows=4)
    with pytest.raises(ValueError, match="samples_valid_count"):
        reader._to_transport_batch(
            {
                "group_id": np.asarray([10, 11], dtype=np.int64),
                "labels": np.asarray([1, 0], dtype=np.int32),
            },
            logical_rank=0,
            logical_world=2,
        )


def test_ranking_reader_rejects_validity_count_above_global_batch() -> None:
    reader = _ranking_reader_for_batch(global_rows=4)

    with pytest.raises(ValueError, match="global batch"):
        reader._to_transport_batch(
            {
                "group_id": np.asarray([10, 11], dtype=np.int64),
                "labels": np.asarray([1, 0], dtype=np.int32),
                "samples_valid_count": np.asarray([5], dtype=np.int32),
            },
            logical_rank=0,
            logical_world=2,
        )
