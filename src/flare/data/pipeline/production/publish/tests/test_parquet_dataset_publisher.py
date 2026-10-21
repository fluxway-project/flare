from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flare.data.connectors.adapters.parquet.sink import ParquetDatasetPublisher


class BatchDataset:
    def __init__(self, batches: list[pa.Table]) -> None:
        self._batches = batches

    def iter_batches(
        self,
        *,
        batch_size: int | None,
        batch_format: str,
        prefetch_batches: int,
    ) -> Iterator[pa.Table]:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "batch_format": batch_format,
            "prefetch_batches": prefetch_batches,
        }
        assert kwargs == {
            "batch_size": None,
            "batch_format": "pyarrow",
            "prefetch_batches": 1,
        }
        yield from self._batches


def test_parquet_dataset_publisher_streams_zstd_partitioned_output(tmp_path) -> None:
    schema = pa.schema(
        [
            pa.field("feature_fid", pa.int64()),
            pa.field("_ds", pa.string()),
            pa.field("_hour", pa.string()),
        ]
    )
    batches = [
        pa.table(
            {
                "feature_fid": [11, 13],
                "_ds": ["2025-12-03", "2025-12-03"],
                "_hour": ["00", "00"],
            },
            schema=schema,
        ),
        pa.table(
            {
                "feature_fid": [17],
                "_ds": ["2025-12-03"],
                "_hour": ["00"],
            },
            schema=schema,
        ),
    ]
    root = tmp_path / "dataset"

    result = ParquetDatasetPublisher(
        root, prefetch_batches=1, max_rows_per_file=2
    ).finalize(BatchDataset(batches), schema=schema)

    assert result.total_rows == 3
    files = tuple(root.rglob("*.parquet"))
    assert len(files) == 2
    assert sum(pq.ParquetFile(path).metadata.num_rows for path in files) == 3
    codecs = {
        pq.ParquetFile(path).metadata.row_group(row_group).column(column).compression
        for path in files
        for row_group in range(pq.ParquetFile(path).metadata.num_row_groups)
        for column in range(
            pq.ParquetFile(path).metadata.row_group(row_group).num_columns
        )
    }
    assert codecs == {"ZSTD"}
    encodings = {
        encoding
        for path in files
        for row_group in range(pq.ParquetFile(path).metadata.num_row_groups)
        for column in range(
            pq.ParquetFile(path).metadata.row_group(row_group).num_columns
        )
        for encoding in pq.ParquetFile(path)
        .metadata.row_group(row_group)
        .column(column)
        .encodings
    }
    assert "RLE_DICTIONARY" not in encodings


def test_parquet_dataset_publisher_appends_across_windows(tmp_path) -> None:
    schema = pa.schema(
        [
            pa.field("fid_101", pa.int64()),
            pa.field("_ds", pa.string()),
            pa.field("_hour", pa.string()),
        ]
    )
    root = tmp_path / "dataset"
    publisher = ParquetDatasetPublisher(
        root,
        prefetch_batches=1,
        max_rows_per_file=2,
    )

    for values in ([11, 13], [17, 19]):
        batch = pa.table(
            {
                "fid_101": values,
                "_ds": ["2025-12-03"] * 2,
                "_hour": ["00"] * 2,
            },
            schema=schema,
        )
        publisher.finalize(BatchDataset([batch]), schema=schema)

    files = tuple(sorted(root.rglob("*.parquet")))
    assert [path.name for path in files] == [
        "part-00000000.parquet",
        "part-00000001.parquet",
    ]
    assert sum(pq.ParquetFile(path).metadata.num_rows for path in files) == 4


def test_parquet_dataset_publisher_overwrite_is_explicit(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    stale = root / "stale.parquet"
    stale.write_bytes(b"stale")

    publisher = ParquetDatasetPublisher(root, overwrite=True)

    assert root.is_dir()
    assert not stale.exists()
    assert publisher.root == root


def test_parquet_dataset_publisher_rejects_existing_output(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(FileExistsError):
        ParquetDatasetPublisher(root)
