from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pytest

from flare.data.pipeline.consumption.source.batching import PyArrowTableOperations
from flare.data.pipeline.consumption.source.batching.stream_reader import finite_batches
from flare.data.pipeline.consumption.source.execution.materialization import (
    coalesce_source_ranges,
    materialize_source_ranges,
)
from flare.data.sources.locator import FileLocator
from flare.data.sources.range import SourceRecordRange


def test_source_record_range_can_merge_and_merge() -> None:
    r1 = SourceRecordRange(
        "s1", "rev1", FileLocator("f1.parquet"), 0, 100, "digest", "parquet"
    )
    r2 = SourceRecordRange(
        "s1", "rev1", FileLocator("f1.parquet"), 100, 200, "digest", "parquet"
    )
    r_gap = SourceRecordRange(
        "s1", "rev1", FileLocator("f1.parquet"), 150, 200, "digest", "parquet"
    )
    r_diff_file = SourceRecordRange(
        "s1", "rev1", FileLocator("f2.parquet"), 100, 200, "digest", "parquet"
    )
    r_diff_rev = SourceRecordRange(
        "s1", "rev2", FileLocator("f1.parquet"), 100, 200, "digest", "parquet"
    )
    r_diff_digest = SourceRecordRange(
        "s1", "rev1", FileLocator("f1.parquet"), 100, 200, "digest2", "parquet"
    )
    r_diff_fmt = SourceRecordRange(
        "s1", "rev1", FileLocator("f1.parquet"), 100, 200, "digest", "csv"
    )

    assert r1.can_merge(r2) is True
    assert r1.can_merge(r_gap) is False
    assert r1.can_merge(r_diff_file) is False
    assert r1.can_merge(r_diff_rev) is False
    assert r1.can_merge(r_diff_digest) is False
    assert r1.can_merge(r_diff_fmt) is False
    assert r1.can_merge(None) is False  # type: ignore[arg-type]

    merged = r1.merge(r2)
    assert merged.record_start == 0
    assert merged.record_count == 300
    assert merged.record_end == 300
    assert merged.locator == FileLocator("f1.parquet")

    with pytest.raises(ValueError, match="cannot merge non-contiguous or incompatible"):
        r1.merge(r_gap)


def test_coalesce_source_ranges_multiple_groups() -> None:
    r1 = SourceRecordRange("s", "r", FileLocator("a.parquet"), 0, 10, "d", "parquet")
    r2 = SourceRecordRange("s", "r", FileLocator("a.parquet"), 10, 20, "d", "parquet")
    r3 = SourceRecordRange("s", "r", FileLocator("b.parquet"), 0, 5, "d", "parquet")
    r4 = SourceRecordRange("s", "r", FileLocator("b.parquet"), 5, 15, "d", "parquet")
    r5 = SourceRecordRange("s", "r", FileLocator("b.parquet"), 20, 10, "d", "parquet")

    plan = coalesce_source_ranges([r1, r2, r3, r4, r5])
    assert len(plan) == 2

    g0_range, g0_slices = plan[0]
    assert g0_range.locator == FileLocator("a.parquet")
    assert g0_range.record_start == 0 and g0_range.record_count == 30
    assert g0_slices == ((0, 0, 10), (1, 10, 20))

    g1_range, g1_slices = plan[1]
    assert g1_range.locator == FileLocator("b.parquet")
    assert g1_range.record_start == 0 and g1_range.record_count == 30
    assert g1_slices == ((2, 0, 5), (3, 5, 15), (4, 20, 10))


def test_materialize_source_ranges_coalesced_execution_and_order_preservation() -> None:
    calls = []

    def mock_read(
        record_range: SourceRecordRange, projection: Sequence[str]
    ) -> pa.Table:
        calls.append(record_range)
        values = list(range(record_range.record_start, record_range.record_end))
        return pa.table({"v": values})

    r0 = SourceRecordRange("s", "r", FileLocator("a.parquet"), 0, 10, "d", "parquet")
    r1 = SourceRecordRange("s", "r", FileLocator("a.parquet"), 10, 10, "d", "parquet")
    r2 = SourceRecordRange("s", "r", FileLocator("b.parquet"), 0, 5, "d", "parquet")
    r3 = SourceRecordRange("s", "r", FileLocator("a.parquet"), 20, 10, "d", "parquet")

    async def scenario():
        return await materialize_source_ranges(
            [r0, r1, r2, r3],
            read=mock_read,
            projection=["v"],
            max_concurrent_reads=4,
        )

    res_table = asyncio.run(scenario())

    assert len(calls) == 3
    assert calls[0].record_count == 20
    assert calls[1].record_count == 5
    assert calls[2].record_count == 10

    expected = list(range(20)) + list(range(5)) + list(range(20, 30))
    assert res_table["v"].to_pylist() == expected


def test_finite_batches_pyarrow_table_assembler_pipeline() -> None:
    c1 = pa.table({"f": list(range(300))})
    c2 = pa.table({"f": list(range(300, 700))})
    c3 = pa.table({"f": list(range(700, 1200))})

    transform_call_counts = 0

    def mock_batch_transformer(tbl: pa.Table) -> dict[str, np.ndarray]:
        nonlocal transform_call_counts
        transform_call_counts += 1
        return {"feat": tbl["f"].to_numpy()}

    batches = list(
        finite_batches(
            iter([c1, c2, c3]),
            batch_size=500,
            completion_policy="drop",
            operations=PyArrowTableOperations(),
            validity_field="samples_valid_count",
        )
    )

    assert len(batches) == 2
    for b in batches:
        assert isinstance(b, pa.Table)
        assert b.num_rows == 500
        transformed = mock_batch_transformer(b)
        assert len(transformed["feat"]) == 500

    assert transform_call_counts == 2
