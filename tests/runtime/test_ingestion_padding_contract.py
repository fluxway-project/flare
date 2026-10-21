from __future__ import annotations

from dataclasses import make_dataclass, replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fluxway.common.ingestion import (
    IngestionLayoutActivation,
    SourceWorkItem,
    WorkerLayoutDescriptor,
)
from fluxway.common.metadata import metadata_digest, pack_metadata
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    LayoutCompileRequest,
    LayoutUnitDescriptor,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
)
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger

from flare.data.connectors.adapters.parquet.source import ParquetSourceAdapter
from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)
from flare.runtime.fluxway.source import AssignedSourceBatchReader
from flare.runtime.fluxway.source import WorkItemScheduledDatasetSourceDescriptor
from flare.runtime.fluxway.workers.reader_contracts import build_reader_source
from flare.data.schema.feature_catalog import FeatureOutputKind
from flare.data.schema.feature_layout import TrainingFeatureLayout
from flare.data.schema.transform import table_to_training_dict
from recipes.recsys.ranking.workers.reader import RankingReaderConfig, RecsysReaderSPMD
from recipes.recsys.ranking.graph.losses import RankingLossAdapter


class ReadCountingParquetSource(ParquetSourceAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_reads = 0

    def read(self, record_range, projection):
        self.data_reads += 1
        return super().read(record_range, projection)


class SingleReaderFactory:
    def __init__(self, reader):
        self.reader = reader

    def create_scan_planner(self, options):
        raise AssertionError("worker materialization must not invoke a planner")

    def create_reader(self, options):
        return self.reader


def compile_tail(record_range=None, *, step=1, batch_size=8):
    if record_range is None:
        record_range = SourceRecordRange(
            source_identity="source",
            revision_identity="revision",
            schema_digest="schema",
            locator=FileLocator("file:///data/part.parquet"),
            record_start=0,
            record_count=5,
            record_format="parquet",
        )
    request = LayoutCompileRequest(
        source_identity=record_range.source_identity,
        view_identity="view",
        schema_digest=record_range.schema_digest,
        lineage_digest="lineage",
        units=(
            LayoutUnitDescriptor(
                unit_id=metadata_digest(record_range.to_dict()),
                revision_identity=record_range.revision_identity,
                cardinality=record_range.record_count,
                cardinality_metric="rows",
                split_policy="SplittableSpan",
                locality_key=record_range.locator.canonical(),
                physical_identity=record_range.locator.canonical(),
                logical_start=record_range.record_start,
                opaque_token=pack_metadata(record_range.to_dict()),
            ),
        ),
        topology=ReaderTopologyDescriptor(
            batch_size,
            tuple(
                ReaderGroupDemand(
                    f"worker-{rank}", (RankDemand(f"rank-{rank}", batch_size // 4),)
                )
                for rank in range(4)
            ),
        ),
        tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
    )
    return AlgebraicLayoutCompiler().compile(request, start_step=step)


def work_items(compiled):
    return tuple(
        SourceWorkItem(
            assignment,
            assignment.descriptor.logical_start,
            assignment.descriptor.logical_start + assignment.descriptor.cardinality,
        )
        for assignment in compiled.work_epoch.assignments
    )


def ranking_reader(*, scheduled=True):
    reader = object.__new__(RecsysReaderSPMD)
    reader.group_id_column = "group_id"
    reader.sparse_groups = frozenset()
    reader._output_type = make_dataclass(
        "PaddingContractOutput",
        [("group_id", object), ("labels", object), ("samples_valid_count", object)],
    )
    reader.ctx = SimpleNamespace(
        step=1,
        logical_rank=0,
        logical_world_size=4,
        source_work_items=work_items(compile_tail()) if scheduled else None,
        global_source_offset=None if scheduled else 0,
    )
    return reader


def ranking_payload():
    return {
        "group_id": np.arange(8, dtype=np.int64),
        "labels": np.zeros(8, dtype=np.int32),
    }


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([5.5]),
        np.asarray([True]),
        np.asarray([5], dtype=np.int64),
        np.asarray([[5]], dtype=np.int32),
    ],
)
def test_ranking_rejects_noncanonical_global_validity(invalid):
    reader = ranking_reader()
    with pytest.raises(ValueError, match="samples_valid_count"):
        reader._to_transport_batch(
            {**ranking_payload(), "samples_valid_count": invalid}, 0, 4
        )


def test_ranking_rejects_scalar_that_conflicts_with_scheduled_work():
    reader = ranking_reader()
    with pytest.raises(ValueError, match="samples_valid_count.*conflict"):
        reader._to_transport_batch(
            {**ranking_payload(), "samples_valid_count": np.asarray([3], np.int32)},
            0,
            4,
        )


def test_ranking_rejects_missing_unscheduled_validity():
    reader = ranking_reader(scheduled=False)
    with pytest.raises(ValueError, match="samples_valid_count"):
        reader._to_transport_batch(ranking_payload(), 0, 4)


def test_ranking_metadata_cannot_override_conflicting_transformed_count():
    reader = ranking_reader()
    reader._batch_transformer = lambda table: {
        **ranking_payload(),
        "samples_valid_count": np.asarray([3], np.int32),
    }
    table = pa.table({"value": range(8)}).replace_schema_metadata(
        {b"samples_valid_count": b"5"}
    )
    with pytest.raises(ValueError, match="samples_valid_count.*conflict"):
        reader.transform_raw_batch(table)


@pytest.mark.asyncio
async def test_pure_padding_preserves_real_parquet_schema_without_data_reads(tmp_path):
    schema = pa.schema([pa.field("feature", pa.list_(pa.int64()), nullable=True)])
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"feature": [[n] for n in range(5)]}, schema=schema), path)
    physical = ReadCountingParquetSource(files=(str(path),), projection=("feature",))
    compiled = compile_tail(physical.record_ranges()[0])
    padding = tuple(item for item in work_items(compiled) if item.is_synthetic_padding)
    session = AssignedSourceBatchReader(reader=physical, projection=("feature",))
    try:
        table = await session.read(1, work_items=padding)
        assert table.schema.remove_metadata() == schema
        assert table["feature"].to_pylist() == [None] * 3
        assert table.schema.metadata[b"samples_valid_count"] == b"5"
        assert physical.data_reads == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_pure_padding_cannot_guess_schema_or_probe_data(tmp_path):
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"value": range(5)}), path)
    physical = ReadCountingParquetSource(files=(str(path),), projection=("value",))
    compiled = compile_tail(physical.record_ranges()[0])
    reader_without_schema = SimpleNamespace(
        bind_source_identity=physical.bind_source_identity,
        read=physical.read,
        read_batches=physical.read_batches,
        close=physical.close,
    )
    session = AssignedSourceBatchReader(
        reader=reader_without_schema, projection=("value",)
    )
    try:
        with pytest.raises(ValueError, match="explicit Arrow schema"):
            await session.read(
                1,
                work_items=tuple(
                    item for item in work_items(compiled) if item.is_synthetic_padding
                ),
            )
        assert physical.data_reads == 0
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "rows", "batch_size"), [(0, 5, 8), (1, 5, 8), (1002, 5559, 8192)]
)
async def test_nested_tail_through_real_ranking_reader_activation(
    tmp_path, step, rows, batch_size
):
    schema = pa.schema(
        [
            pa.field("fid_1", pa.list_(pa.int64()), nullable=False),
            pa.field("fid_2", pa.list_(pa.int32()), nullable=False),
            pa.field("label", pa.int32(), nullable=False),
            pa.field("group_id", pa.int64(), nullable=False),
        ],
        metadata={b"contract": b"nested-ranking"},
    )
    path = tmp_path / "ranking.parquet"
    pq.write_table(
        pa.table(
            {
                "fid_1": [[n + 1] for n in range(rows)],
                "fid_2": [[1, 2]] * rows,
                "label": [n % 2 for n in range(rows)],
                "group_id": list(range(rows)),
            },
            schema=schema,
        ),
        path,
    )
    layout = TrainingFeatureLayout(
        projection=schema.names,
        bindings=(
            ("feature", "fid_1"),
            ("side", "fid_2"),
            ("label", "label"),
            ("group_id", "group_id"),
        ),
        kinds_by_name={
            "feature": FeatureOutputKind.SPARSE_I64_FIDS,
            "side": FeatureOutputKind.DENSE_I32_INDICES,
            "label": FeatureOutputKind.DENSE_I32_INDICES,
            "group_id": FeatureOutputKind.SPARSE_I64_FIDS,
        },
        padding_by_name={"side": 0},
    )
    physical = ReadCountingParquetSource(
        files=(str(path),), projection=layout.projection
    )
    compiled = compile_tail(
        physical.record_ranges()[0], step=step, batch_size=batch_size
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    ledger.restore_state({"committed_cut": step})
    descriptors = compiled.encoded_worker_layouts()
    await ledger.publish(
        compiled.work_epoch,
        tuple(WorkerLayoutDescriptor.decode(value) for value in descriptors),
    )
    output_type = make_dataclass(
        "NestedRankingOutput",
        [
            (name, object)
            for name in (
                "embedding",
                "side_info",
                "labels",
                "group_id",
                "samples_valid_count",
            )
        ],
    )
    counts = []
    for rank, descriptor in enumerate(descriptors):
        physical = ReadCountingParquetSource(
            files=(str(path),), projection=layout.projection
        )
        source = build_reader_source(
            descriptor=WorkItemScheduledDatasetSourceDescriptor(
                source_factory=SingleReaderFactory(physical),
                global_batch_size=batch_size,
                completion_policy="pad",
            ),
            projection=layout.projection,
            source_read_concurrency=2,
            storage_options={},
        )
        reader = RecsysReaderSPMD(
            reader_config=RankingReaderConfig(
                source=source,
                feature_layout=layout,
                sparse_groups=frozenset({"embedding"}),
                transform=partial(
                    table_to_training_dict,
                    side_info_seq=(("side", 2),),
                    categorical_feature_groups={"embedding": ("feature",)},
                    metadata_cols=("group_id",),
                    label_column="label",
                    feature_layout=layout,
                ),
                output_type=output_type,
            )
        )
        leases = await ledger.acquire(0, compiled.worker_layouts[rank].segments)
        reader.ctx = SimpleNamespace(
            step=step,
            logical_rank=rank,
            logical_world_size=4,
            source_work_items=tuple(lease.item for lease in leases),
        )
        try:
            await reader.activate_ingestion_layout(
                IngestionLayoutActivation(
                    epoch=0,
                    start_position=step,
                    end_position=step + 1,
                    layout_digest=compiled.layout_digest,
                    terminal=True,
                    descriptor=descriptor,
                )
            )
            output = await reader.read()
            counts.append(output.samples_valid_count.flat_values.item())
            assert output.labels.flat_values.dtype == np.int32
            assert output.side_info.global_shape == (batch_size, 2)
            if rank == 3:
                assert physical.data_reads == 0
                assert output.embedding.flat_values.tolist() == [
                    layout.null_index_sentinel
                ] * (batch_size // 4)
                loss = RankingLossAdapter(
                    lambda logits, labels, weights: jnp.sum(weights)
                )
                contribution = loss(
                    {"logits": jnp.zeros((batch_size // 4, 1))},
                    {
                        "labels": jnp.asarray(output.labels.flat_values),
                        "samples_valid_count": jnp.asarray(
                            output.samples_valid_count.flat_values
                        ),
                        "global_row_start": jnp.asarray(rank * (batch_size // 4)),
                        "world_size": jnp.asarray(4),
                    },
                )
                assert float(contribution) == 0.0
            else:
                assert physical.data_reads == 1
        finally:
            if reader._assigned_session is not None:
                await reader._assigned_session.close()
    assert counts == [rows] * 4


@pytest.mark.asyncio
async def test_pure_padding_still_binds_source_identity(tmp_path):
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"value": range(5)}), path)
    physical = ParquetSourceAdapter(files=(str(path),), projection=("value",))
    compiled = compile_tail(physical.record_ranges()[0])
    padding = tuple(item for item in work_items(compiled) if item.is_synthetic_padding)
    session = AssignedSourceBatchReader(reader=physical, projection=("value",))
    try:
        await session.read(1, work_items=padding)
        assert session.bound_source_identity == physical.source_identity
        changed = compile_tail(
            replace(physical.record_ranges()[0], source_identity="replacement"), step=2
        )
        with pytest.raises(ValueError, match="identity changed"):
            await session.read(
                2,
                work_items=tuple(
                    item for item in work_items(changed) if item.is_synthetic_padding
                ),
            )
    finally:
        await session.close()
