from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa


def read_record_batch_range(
    batches: Iterable[pa.RecordBatch], *, row_start: int, row_count: int
) -> pa.Table:
    if row_start < 0 or row_count <= 0:
        raise ValueError(
            "record range must be non-negative and non-empty: "
            f"start={row_start} count={row_count}"
        )
    skip = row_start
    remaining = row_count
    selected: list[pa.RecordBatch] = []
    for batch in batches:
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(
                "streaming source reader must yield pyarrow.RecordBatch values, "
                f"got {type(batch).__name__}"
            )
        if skip >= batch.num_rows:
            skip -= batch.num_rows
            continue
        start = skip
        take = min(batch.num_rows - start, remaining)
        selected.append(batch.slice(start, take))
        remaining -= take
        skip = 0
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"record range exceeds source rows: start={row_start} count={row_count}"
        )
    return pa.Table.from_batches(selected)
