from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import pyarrow as pa

from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange
from flare.data.pipeline.consumption.source.execution.invocation import invoke_source

DEFAULT_SOURCE_READ_CONCURRENCY = 2


def validate_source_read_concurrency(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("source read concurrency must be a positive integer")
    return value


def coalesce_source_ranges(
    ranges: Sequence[SourceRecordRange],
) -> tuple[tuple[SourceRecordRange, tuple[tuple[int, int, int], ...]], ...]:
    """Group contiguous ranges by source identity and file locator.

    Returns a tuple of (execution_range, slices_mapping), where:
    - execution_range is the coalesced single SourceRecordRange to read.
    - slices_mapping is a tuple of (original_index, offset_within_execution_range, row_count).
    """
    if not ranges:
        return ()

    execution_groups: list[list[tuple[int, SourceRecordRange]]] = []
    current_group: list[tuple[int, SourceRecordRange]] = [(0, ranges[0])]

    for idx in range(1, len(ranges)):
        r = ranges[idx]
        _prev_idx, prev_r = current_group[-1]
        if prev_r.can_merge(r):
            current_group.append((idx, r))
        else:
            execution_groups.append(current_group)
            current_group = [(idx, r)]
    execution_groups.append(current_group)

    result = []
    for group in execution_groups:
        _first_idx, first_r = group[0]
        merged = first_r
        for _, r in group[1:]:
            merged = merged.merge(r)
        offset = 0
        slices = []
        for orig_idx, r in group:
            slices.append((orig_idx, offset, r.record_count))
            offset += r.record_count
        result.append((merged, tuple(slices)))

    return tuple(result)


async def materialize_source_ranges(
    ranges: Sequence[SourceRecordRange],
    *,
    read: Callable[[SourceRecordRange, Sequence[str]], Any],
    projection: Sequence[str],
    max_concurrent_reads: int,
) -> Any:
    if not ranges:
        raise ValueError("source ranges must not be empty")
    max_concurrent_reads = validate_source_read_concurrency(max_concurrent_reads)
    coalesced_plan = coalesce_source_ranges(ranges)

    if len(coalesced_plan) == 1:
        merged_range, _slices = coalesced_plan[0]
        result = await invoke_source(read, merged_range, projection)
        if not isinstance(result, pa.Table):
            raise TypeError(
                "source range materialization requires pyarrow.Table, "
                f"got {type(result).__name__}"
            )
        return result

    missing = object()
    payloads: list[Any] = [missing] * len(ranges)
    pending = iter(enumerate(coalesced_plan))

    async def worker() -> None:
        while True:
            try:
                _group_idx, (merged_range, slices) = next(pending)
            except StopIteration:
                return
            result = await invoke_source(read, merged_range, projection)
            if not isinstance(result, pa.Table):
                raise TypeError(
                    "source range materialization requires pyarrow.Table, "
                    f"got {type(result).__name__}"
                )
            if len(slices) == 1:
                orig_idx, _, _ = slices[0]
                payloads[orig_idx] = result
            else:
                for orig_idx, offset, count in slices:
                    if isinstance(result, pa.Table):
                        payloads[orig_idx] = result.slice(offset, count)
                    else:
                        raise TypeError(
                            "source range materialization requires pyarrow.Table, "
                            f"got {type(result).__name__}"
                        )

    tasks = [
        asyncio.create_task(worker())
        for _ in range(min(max_concurrent_reads, len(coalesced_plan)))
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if any(payload is missing for payload in payloads):
        raise RuntimeError("source range worker completed without a payload")
    return concatenate_range_payloads(payloads)


def concatenate_range_payloads(payloads: Sequence[Any]) -> Any:
    if not payloads:
        raise ValueError("range payloads must not be empty")
    first = payloads[0]
    if len(payloads) == 1:
        return first
    if isinstance(first, pa.Table):
        if any(not isinstance(item, pa.Table) for item in payloads[1:]):
            raise TypeError("range payloads must all be pyarrow.Table values")
        return pa.concat_tables(payloads, promote_options="permissive")
    raise TypeError(
        "source range materialization requires pyarrow.Table, "
        f"got {type(first).__name__}"
    )
