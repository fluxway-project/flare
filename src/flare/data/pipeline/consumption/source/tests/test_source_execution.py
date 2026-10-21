import asyncio
import threading

from flare.data.pipeline.consumption.source.execution.invocation import invoke_source


def test_sync_source_call_runs_off_event_loop() -> None:
    event_loop_thread = threading.get_ident()

    def read(value: int) -> tuple[int, int]:
        return value, threading.get_ident()

    async def scenario() -> None:
        value, worker_thread = await invoke_source(read, 3)
        assert value == 3
        assert worker_thread != event_loop_thread

    asyncio.run(scenario())


def test_async_source_call_is_awaited_without_duplicate_execution() -> None:
    calls = []

    async def discover(value: int) -> int:
        calls.append(value)
        await asyncio.sleep(0)
        return value + 1

    async def scenario() -> None:
        assert await invoke_source(discover, 4) == 5
        assert calls == [4]

    asyncio.run(scenario())


def test_sync_source_method_returning_awaitable_is_awaited() -> None:
    async def result() -> str:
        return "ready"

    def discover() -> object:
        return result()

    async def scenario() -> None:
        assert await invoke_source(discover) == "ready"

    asyncio.run(scenario())


def test_coalesce_source_ranges_groups_contiguous_slices() -> None:
    from flare.data.pipeline.consumption.source.execution.materialization import (
        coalesce_source_ranges,
    )
    from flare.data.sources.locator import FileLocator
    from flare.data.sources.range import SourceRecordRange

    r1 = SourceRecordRange("s", "r", FileLocator("f1.parquet"), 0, 10, "d", "parquet")
    r2 = SourceRecordRange("s", "r", FileLocator("f1.parquet"), 10, 20, "d", "parquet")
    r3 = SourceRecordRange("s", "r", FileLocator("f2.parquet"), 0, 5, "d", "parquet")
    r4 = SourceRecordRange("s", "r", FileLocator("f2.parquet"), 5, 15, "d", "parquet")
    r5 = SourceRecordRange("s", "r", FileLocator("f2.parquet"), 30, 10, "d", "parquet")

    plan = coalesce_source_ranges([r1, r2, r3, r4, r5])
    assert len(plan) == 3

    # Group 0: f1 0..30
    g0_range, g0_slices = plan[0]
    assert g0_range.locator == FileLocator("f1.parquet")
    assert g0_range.record_start == 0 and g0_range.record_count == 30
    assert g0_slices == ((0, 0, 10), (1, 10, 20))

    # Group 1: f2 0..20
    g1_range, g1_slices = plan[1]
    assert g1_range.locator == FileLocator("f2.parquet")
    assert g1_range.record_start == 0 and g1_range.record_count == 20
    assert g1_slices == ((2, 0, 5), (3, 5, 15))

    # Group 2: f2 30..40 (gap)
    g2_range, g2_slices = plan[2]
    assert g2_range.locator == FileLocator("f2.parquet")
    assert g2_range.record_start == 30 and g2_range.record_count == 10
    assert g2_slices == ((4, 0, 10),)


def test_materialize_source_ranges_coalesces_reads_and_preserves_order() -> None:
    import pyarrow as pa

    from flare.data.pipeline.consumption.source.execution.materialization import (
        materialize_source_ranges,
    )
    from flare.data.sources.locator import FileLocator
    from flare.data.sources.range import SourceRecordRange

    read_calls = []

    def mock_read(record_range: SourceRecordRange, projection):
        read_calls.append(record_range)
        values = list(
            range(
                record_range.record_start,
                record_range.record_start + record_range.record_count,
            )
        )
        return pa.table({"val": values})

    r1 = SourceRecordRange("s", "r", FileLocator("f1.parquet"), 0, 20, "d", "parquet")
    r2 = SourceRecordRange("s", "r", FileLocator("f1.parquet"), 20, 30, "d", "parquet")
    r3 = SourceRecordRange("s", "r", FileLocator("f2.parquet"), 100, 15, "d", "parquet")
    r4 = SourceRecordRange("s", "r", FileLocator("f2.parquet"), 115, 25, "d", "parquet")

    async def scenario():
        return await materialize_source_ranges(
            [r1, r2, r3, r4],
            read=mock_read,
            projection=["val"],
            max_concurrent_reads=4,
        )

    res_table = asyncio.run(scenario())

    # 4 contiguous ranges coalesced into exactly 2 reads
    assert len(read_calls) == 2
    assert (
        read_calls[0].locator == FileLocator("f1.parquet")
        and read_calls[0].record_count == 50
    )
    assert (
        read_calls[1].locator == FileLocator("f2.parquet")
        and read_calls[1].record_count == 40
    )

    # Data content and ordering strictly preserved
    expected = list(range(50)) + list(range(100, 140))
    assert res_table["val"].to_pylist() == expected


def test_materialize_source_ranges_single_coalesced_fast_path() -> None:
    import pyarrow as pa

    from flare.data.pipeline.consumption.source.execution.materialization import (
        materialize_source_ranges,
    )
    from flare.data.sources.locator import FileLocator
    from flare.data.sources.range import SourceRecordRange

    raw_table = pa.table({"x": [1, 2, 3, 4]})

    def mock_read(record_range: SourceRecordRange, projection):
        return raw_table

    r1 = SourceRecordRange("s", "r", FileLocator("f1.parquet"), 0, 2, "d", "parquet")
    r2 = SourceRecordRange("s", "r", FileLocator("f1.parquet"), 2, 2, "d", "parquet")

    async def scenario():
        return await materialize_source_ranges(
            [r1, r2],
            read=mock_read,
            projection=["x"],
            max_concurrent_reads=2,
        )

    res = asyncio.run(scenario())
    # Single coalesced range returns the exact object directly
    assert res is raw_table
