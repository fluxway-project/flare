from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flare.data.pipeline.consumption.dataloader import ParquetRangeBatchLoader


def test_parquet_training_loader_owns_static_ranges_and_transform(tmp_path) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3, 4]}), path)

    loader = ParquetRangeBatchLoader(
        dataset_path=str(path),
        projection=("value",),
        batch_size=2,
        batch_transformer=lambda table: tuple(table["value"].to_pylist()),
        num_epochs=2,
    )

    assert list(loader) == [(1, 2), (3, 4), (1, 2), (3, 4)]


def test_parquet_training_loader_shards_ranges(tmp_path) -> None:
    pq.write_table(pa.table({"value": [1, 2]}), tmp_path / "part-0.parquet")
    pq.write_table(pa.table({"value": [3, 4]}), tmp_path / "part-1.parquet")

    loader = ParquetRangeBatchLoader(
        dataset_path=str(tmp_path),
        projection=("value",),
        batch_size=4,
        batch_transformer=lambda table: table["value"].to_pylist(),
        shard_index=1,
        shard_count=2,
    )

    assert list(loader) == [[3, 4]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"shard_count": 0},
        {"shard_index": 1, "shard_count": 1},
        {"num_epochs": 0},
    ],
)
def test_parquet_training_loader_rejects_invalid_bounds(tmp_path, kwargs) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(pa.table({"value": [1]}), path)

    options = {
        "dataset_path": str(path),
        "projection": ("value",),
        "batch_size": 1,
        "batch_transformer": lambda table: table,
    }
    options.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ParquetRangeBatchLoader(**options)
