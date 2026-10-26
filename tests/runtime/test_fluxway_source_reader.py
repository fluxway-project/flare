from __future__ import annotations

import asyncio
import threading
from builtins import ExceptionGroup
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import make_dataclass, replace
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

from flare.data.contracts.assets import AssetRef
from flare.data.contracts.lineage import LineageConfig, LineageContext
from flare.data.pipeline.consumption.bootstrap.factory import StandardSourceFactory
from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)
from flare.data.pipeline.consumption.source.contracts.record_range import (
    RecordRangeReader,
)
from flare.data.pipeline.consumption.source.contracts.source import SourceBatch
from flare.data.pipeline.consumption.source.execution.invocation import (
    read_source_batch,
)
from flare.data.pipeline.consumption.source.execution.record_ranges import (
    RecordRangeReadSession,
)
from flare.runtime.fluxway.source import (
    WorkItemScheduledDatasetSourceDescriptor,
)
from flare.runtime.fluxway.workers.reader import BatchReaderSPMD
from flare.runtime.fluxway.workers.reader_contracts import build_reader_source
from recipes.recsys.ranking.workers.reader import RecsysReaderSPMD


class DescriptorFactory:
    def prepare_options(self, options):
        return dict(options)

    def create_scan_planner(self, options):
        raise AssertionError("planner must not be called")

    def create_reader(self, options, *, include_lineage=False):
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


def test_standard_reader_requires_projection() -> None:
    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=StandardSourceFactory(),
        global_batch_size=2,
        completion_policy="drop",
    )

    with pytest.raises(TypeError, match="projection"):
        build_reader_source(
            descriptor=descriptor,
            source_read_concurrency=1,
            storage_options={"storage_format": "parquet"},
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
            "catalog_name": "test_metalake",
            "catalog_type": "rest",
            "catalog_uri": "http://catalog.example/iceberg",
            "table_identifier": "throughput.input_graph",
            "catalog_token": "test-token",
        },
    )
    assert source.options["catalog_token"] == "test-token"


class DummyRangeReader(RecordRangeReader):
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
    source = DummyRangeReader()
    reader = RecordRangeReadSession(
        reader=source,
        projection=("value",),
    )

    result = (
        await read_source_batch(
            reader,
            0,
            work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        )
    ).payload

    assert isinstance(result, pa.Table)
    assert result["value"].to_pylist() == [12, 13, 14, 15, 16]
    assert source.read_calls[0].record_start == 12
    assert source.read_calls[0].record_count == 5
    await asyncio.gather(reader.close(), reader.close())
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_assigned_reader_reuses_range_decode_across_steps(monkeypatch) -> None:
    import flare.data.pipeline.consumption.source.execution.record_ranges as ranges

    reader = RecordRangeReadSession(reader=DummyRangeReader(), projection=("value",))
    unpack = ranges.unpack_metadata
    decoded = []

    def count_decode(token):
        decoded.append(token)
        return unpack(token)

    monkeypatch.setattr(ranges, "unpack_metadata", count_decode)
    try:
        first = await read_source_batch(
            reader,
            0,
            work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        )
        second = await read_source_batch(
            reader,
            1,
            work_items=(_work_item(step=1, logical_start=17, logical_end=22),),
        )
        assert first.payload["value"].to_pylist() == [12, 13, 14, 15, 16]
        assert second.payload["value"].to_pylist() == [17, 18, 19, 20, 21]
        assert len(decoded) == 1
    finally:
        await reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["schema_digest", "revision_identity", "physical_identity"]
)
async def test_cached_range_still_rejects_changed_assignment(field) -> None:
    source = DummyRangeReader()
    reader = RecordRangeReadSession(reader=source, projection=("value",))
    work = _work_item(step=0, logical_start=12, logical_end=17)
    try:
        await read_source_batch(reader, 0, work_items=(work,))
        if field == "schema_digest":
            changed = replace(work.assignment, schema_digest="replacement")
        else:
            changed = replace(
                work.assignment,
                descriptor=replace(
                    work.assignment.descriptor, **{field: "replacement"}
                ),
            )
        with pytest.raises(ValueError, match="conflicts with assignment"):
            await read_source_batch(
                reader, 0, work_items=(replace(work, assignment=changed),)
            )
        assert len(source.read_calls) == 1
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_assigned_reader_binds_identity_once_and_rejects_changes() -> None:
    source = DummyRangeReader()
    reader = RecordRangeReadSession(reader=source, projection=("value",))

    batch = await read_source_batch(
        reader,
        0,
        work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
    )
    assert batch.payload is not None
    assert source.bound_source_identity == "source"

    with pytest.raises(ValueError, match="identity changed"):
        await read_source_batch(
            reader,
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
        def prepare_options(self, options):
            return dict(options)

        def create_scan_planner(self, options):
            raise AssertionError("planner must not be called")

        def create_reader(self, options, *, include_lineage=False):
            return RecordRangeReadSession(
                reader=DummyRangeReader(),
                projection=options["projection"],
            )

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
    reader = RecordRangeReadSession(
        reader=DummyRangeReader(),
        projection=("value",),
    )

    with pytest.raises(ValueError, match="does not match the requested step"):
        await read_source_batch(
            reader,
            1,
            work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        )


@pytest.mark.asyncio
async def test_assigned_reader_resolves_slice_from_original_file_range() -> None:
    source = DummyRangeReader()
    reader = RecordRangeReadSession(reader=source, projection=("value",))

    result = (
        await read_source_batch(
            reader,
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
    ).payload

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
    source = DummyRangeReader()
    reader = RecordRangeReadSession(reader=source, projection=("value",))
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
            batch = (
                await read_source_batch(reader, step, work_items=work_items)
            ).payload
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


@pytest.mark.asyncio
async def test_native_query_session_preserves_async_payload_and_opaque_tasks():
    original = _work_item(step=0, logical_start=12, logical_end=17)
    unit = replace(
        original.assignment.descriptor,
        unit_id="query-partition",
        locality_key="query-partition",
        physical_identity="query://partition",
        opaque_token=b"native-query-token",
    )
    assignment = SourceWorkAssignment.create(
        **{
            name: getattr(original.assignment, name)
            for name in (
                "epoch",
                "step",
                "source_identity",
                "view_identity",
                "schema_digest",
                "lineage_digest",
                "global_start",
                "global_batch_size",
                "global_valid_count",
            )
        },
        descriptor=unit,
    )
    work = SourceWorkItem(assignment, original.logical_start, original.logical_end)
    payload = {"value": np.arange(work.cardinality, dtype=np.int32)}
    loop = asyncio.get_running_loop()

    class NativeSession:
        closes = 0

        async def read(self, step, *, work_items):
            assert asyncio.get_running_loop() is loop
            assert (
                work_items[0].assignment.descriptor.opaque_token
                == b"native-query-token"
            )
            return SourceBatch(
                payload, work.cardinality, (AssetRef("query", "dataset", "v1"),)
            )

        async def close(self):
            self.closes += 1

    native = NativeSession()

    class NativeFactory:
        def prepare_options(self, options):
            assert set(options) == {"native_query"}
            if not isinstance(options.get("native_query"), str):
                raise TypeError("native_query must be a string")
            return dict(options)

        async def create_scan_planner(self, options):
            raise AssertionError("read execution must not invoke planning")

        async def create_reader(self, options, *, include_lineage=False):
            assert asyncio.get_running_loop() is loop
            assert set(options) == {"native_query"}
            assert options["native_query"] == "select-native"
            return native

    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=NativeFactory(), global_batch_size=20, completion_policy="drop"
    )
    source = build_reader_source(
        descriptor=descriptor,
        storage_options={"native_query": "select-native"},
    )

    class NativeReader(BatchReaderSPMD):
        def transform_raw_batch(self, value):
            assert value is payload
            assert value["value"] is payload["value"]
            return value

    reader = NativeReader(source=source, output_builder=dict)
    await reader.activate_ingestion_layout(_activation())
    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=(work,),
        global_source_offset=0,
    )
    result = await reader.read()
    assert "value" in result
    await reader.close()
    assert native.closes == 1


@pytest.mark.asyncio
async def test_source_protocol_rejects_incorrect_native_cardinality():
    class NativeSession:
        def read(self, step, *, work_items):
            return SourceBatch({"native": object()}, 1)

        def close(self):
            pass

    with pytest.raises(ValueError, match="cardinality"):
        await read_source_batch(
            NativeSession(),
            0,
            work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("audit", [False, True])
async def test_reader_uses_existing_profiler_and_audit_switches(
    monkeypatch, tmp_path, profile, audit
):
    from fluxway import profiler

    scopes = []

    @contextmanager
    def scope(name):
        assert profile, "disabled profiling must not enter scope wrappers"
        scopes.append(name)
        yield

    monkeypatch.setattr(profiler, "enabled", lambda: profile)
    monkeypatch.setattr(profiler, "scope", scope)

    class Factory(DescriptorFactory):
        def create_reader(self, options, *, include_lineage=False):
            assert include_lineage is False
            return RecordRangeReadSession(
                reader=DummyRangeReader(), projection=("value",)
            )

    audit_path = tmp_path / "audit.json"
    reader = BatchReaderSPMD(
        source=build_reader_source(
            descriptor=WorkItemScheduledDatasetSourceDescriptor(
                source_factory=Factory(), global_batch_size=20, completion_policy="drop"
            )
        ),
        output_builder=dict,
        execution_audit_path=str(audit_path) if audit else None,
    )
    reader.rank = 0
    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=(_work_item(step=0, logical_start=12, logical_end=17),),
        global_source_offset=0,
    )
    await reader.activate_ingestion_layout(_activation())
    try:
        result = await reader.read()
        assert result["value"].flat_values.tolist() == [12, 13, 14, 15, 16]
    finally:
        await reader.on_shutdown()
    assert scopes == (
        [
            "sourcework.validate_assignments",
            "sourcework.materialize_ranges",
            "reader.to_transport_batch",
        ]
        if profile
        else []
    )
    rank_path = tmp_path / "audit_rank_0.json"
    if audit:
        import json

        records = json.loads(rank_path.read_text())["records"]
        assert len(records) == 1
        assert records[0]["logical_start"] == 12
        assert records[0]["cardinality"] == 5
    else:
        assert not rank_path.exists()
        assert reader._execution_audit_records is None


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_cancelled_activation_keeps_session_creation_owned_until_close(
    synchronous,
):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    async_release = asyncio.Event()
    sync_release = threading.Event()

    class Session:
        closes = 0

        def read(self, step, *, work_items):
            raise AssertionError("creation lifecycle must not read data")

        async def close(self):
            self.closes += 1

    session = Session()

    async def async_create(options, *, include_lineage=False):
        assert asyncio.get_running_loop() is loop
        entered.set()
        await async_release.wait()
        return session

    def sync_create(options, *, include_lineage=False):
        loop.call_soon_threadsafe(entered.set)
        assert sync_release.wait(timeout=5)
        return session

    class Factory(DescriptorFactory):
        create_reader = staticmethod(sync_create if synchronous else async_create)

    reader = BatchReaderSPMD(
        source=build_reader_source(
            descriptor=WorkItemScheduledDatasetSourceDescriptor(
                source_factory=Factory(), global_batch_size=2, completion_policy="drop"
            )
        ),
        output_builder=dict,
    )
    activation = asyncio.create_task(reader.activate_ingestion_layout(_activation()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        activation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await activation
        closing = asyncio.create_task(reader.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert session.closes == 0
    finally:
        async_release.set()
        sync_release.set()
        await reader.close()
        await asyncio.gather(activation, return_exceptions=True)
    await closing
    assert session.closes == 1


@pytest.mark.asyncio
async def test_cancelled_reader_close_keeps_one_cleanup_and_rejects_activation():
    entered, release = asyncio.Event(), asyncio.Event()

    class Session:
        closes = 0

        def read(self, step, *, work_items):
            raise AssertionError("close lifecycle must not read data")

        async def close(self):
            self.closes += 1
            entered.set()
            await release.wait()

    session = Session()

    class Factory(DescriptorFactory):
        def create_reader(self, options, *, include_lineage=False):
            return session

    reader = BatchReaderSPMD(
        source=build_reader_source(
            descriptor=WorkItemScheduledDatasetSourceDescriptor(
                source_factory=Factory(), global_batch_size=2, completion_policy="drop"
            )
        ),
        output_builder=dict,
    )
    await reader.activate_ingestion_layout(_activation())
    closing = asyncio.create_task(reader.close())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(RuntimeError, match="closed"):
            await reader.activate_ingestion_layout(_activation())
        waiting = asyncio.create_task(reader.close())
        await asyncio.sleep(0)
        assert not waiting.done()
    finally:
        release.set()
        await reader.close()
    await waiting
    assert session.closes == 1


@pytest.mark.asyncio
async def test_lazy_lineage_creation_failure_does_not_prevent_source_cleanup():
    work = _work_item(step=0, logical_start=12, logical_end=17)

    class Session:
        closes = 0

        def read(self, step, *, work_items):
            return SourceBatch(
                {"value": np.arange(work.cardinality)},
                work.cardinality,
                (AssetRef("native", "dataset"),),
            )

        async def close(self):
            self.closes += 1
            raise RuntimeError("source cleanup failed")

    session = Session()

    class Factory(DescriptorFactory):
        def create_reader(self, options, *, include_lineage=False):
            return session

    class ExporterFactory:
        calls = 0

        async def create(self, context):
            self.calls += 1
            raise ValueError("lineage creation failed")

    exporter_factory = ExporterFactory()
    reader = BatchReaderSPMD(
        source=build_reader_source(
            descriptor=WorkItemScheduledDatasetSourceDescriptor(
                source_factory=Factory(),
                global_batch_size=2,
                completion_policy="drop",
                lineage=LineageConfig(
                    exporter_factory, LineageContext("run", "operation")
                ),
            )
        ),
        output_builder=dict,
    )
    await reader.activate_ingestion_layout(_activation())
    assert exporter_factory.calls == 0
    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=(work,),
        global_source_offset=0,
    )
    await reader.read()
    assert exporter_factory.calls == 0
    with pytest.raises(ExceptionGroup, match="source reader close failed") as raised:
        await reader.close()
    assert str(raised.value.exceptions[0]) == "source cleanup failed"
    assert str(raised.value.exceptions[1].exceptions[0]) == "lineage creation failed"
    assert exporter_factory.calls == 1
    assert session.closes == 1


def test_default_reader_accepts_native_arrow_batch_without_copying_numeric_values():
    reader = BatchReaderSPMD(
        source=build_reader_source(
            read_session=LocalSegmentReader(), read_chunk_size=2
        ),
        output_builder=dict,
    )
    values = pa.array([3, 5, 7], type=pa.int32())
    raw = pa.record_batch([values], names=["value"])
    columns = reader.transform_raw_batch(raw)
    assert (
        columns["value"].__array_interface__["data"][0] == values.buffers()[1].address
    )


@pytest.mark.parametrize("values", [[True, False], [1, None], ["a", "b"]])
def test_arrow_table_and_record_batch_use_the_same_consumer_conversion(values):
    reader = BatchReaderSPMD(
        source=build_reader_source(
            read_session=LocalSegmentReader(), read_chunk_size=2
        ),
        output_builder=dict,
    )
    batch = pa.record_batch([pa.array(values)], names=["value"])
    table = pa.Table.from_batches([batch])
    np.testing.assert_equal(
        reader.transform_raw_batch(table)["value"],
        reader.transform_raw_batch(batch)["value"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_cancelled_read_is_drained_before_releasing_session(synchronous):
    loop = asyncio.get_running_loop()
    entered, async_release = asyncio.Event(), asyncio.Event()
    sync_release = threading.Event()
    work = _work_item(step=0, logical_start=12, logical_end=17)

    class Session:
        closes = 0
        running = False

        async def read_async(self, step, *, work_items):
            self.running = True
            entered.set()
            await async_release.wait()
            self.running = False
            return SourceBatch({"value": np.arange(work.cardinality)}, work.cardinality)

        def read_sync(self, step, *, work_items):
            self.running = True
            loop.call_soon_threadsafe(entered.set)
            assert sync_release.wait(timeout=5)
            self.running = False
            return SourceBatch({"value": np.arange(work.cardinality)}, work.cardinality)

        read = read_sync if synchronous else read_async

        def close(self):
            assert not self.running
            self.closes += 1

    session = Session()

    class Factory(DescriptorFactory):
        def create_reader(self, options, *, include_lineage=False):
            return session

    reader = BatchReaderSPMD(
        source=build_reader_source(
            descriptor=WorkItemScheduledDatasetSourceDescriptor(
                source_factory=Factory(), global_batch_size=20, completion_policy="drop"
            )
        ),
        output_builder=dict,
    )
    await reader.activate_ingestion_layout(_activation())
    reader.ctx = SimpleNamespace(
        step=0,
        logical_rank=0,
        logical_world_size=1,
        source_work_items=(work,),
        global_source_offset=0,
    )
    reading = asyncio.create_task(reader.read())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        closing = asyncio.create_task(reader.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert session.closes == 0
    finally:
        async_release.set()
        sync_release.set()
        await reader.close()
    await closing
    assert session.closes == 1
