"""Benchmark Iceberg Table.io reads against explicit direct S3 reads."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem

from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    load_iceberg_table,
)
from tools.validation.performance_metrics import Measurement, summarize, write_json


@dataclass(frozen=True)
class FileRange:
    path: str
    rows: int


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required for the direct S3 benchmark")
    return value


def planned_ranges(table) -> tuple[FileRange, ...]:
    ranges = []
    for task in table.scan().plan_files():
        if task.delete_files:
            raise NotImplementedError("benchmark does not support delete files")
        ranges.append(FileRange(str(task.file.file_path), int(task.file.record_count)))
    if not ranges:
        raise RuntimeError("Iceberg table has no data files")
    return tuple(ranges)


def projected_columns(table, count: int) -> tuple[str, ...]:
    columns = tuple(table.schema().columns)
    if count <= 0 or count > len(columns):
        raise ValueError(f"projection count must be between 1 and {len(columns)}")
    return tuple(column.name for column in columns[:count])


def read_via_table_io(
    table, file_range: FileRange, columns: tuple[str, ...]
) -> pa.Table:
    with table.io.new_input(file_range.path).open() as stream:
        return pq.read_table(stream, columns=list(columns))


def read_via_s3(
    filesystem: S3FileSystem, file_range: FileRange, columns: tuple[str, ...]
) -> pa.Table:
    parsed = urlparse(file_range.path)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            f"direct S3 benchmark requires an s3:// file path: {file_range.path}"
        )
    with filesystem.open_input_file(f"{parsed.netloc}{parsed.path}") as stream:
        return pq.read_table(stream, columns=list(columns))


def parallel_read(read_one, ranges, columns, concurrency: int) -> tuple[pa.Table, ...]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return tuple(executor.map(lambda item: read_one(item, columns), ranges))


def benchmark(label, read_one, ranges, columns, repetitions, concurrency) -> dict:
    measurements = []
    reference = None
    for _ in range(repetitions):
        started = time.perf_counter()
        tables = parallel_read(read_one, ranges, columns, concurrency)
        elapsed = time.perf_counter() - started
        current = pa.Table.from_batches(
            [batch for table in tables for batch in table.to_batches()]
        )
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise AssertionError(f"{label} returned different Arrow data")
        measurements.append(Measurement(elapsed, sum(item.rows for item in ranges)))
    return {"label": label, **summarize(measurements)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_identifier")
    parser.add_argument("--rest-uri", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--projection-count", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--output", type=__import__("pathlib").Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    table = load_iceberg_table(
        build_iceberg_catalog_kwargs(
            name="source-read-benchmark",
            catalog_type="rest",
            uri=args.rest_uri,
            warehouse=args.warehouse,
        ),
        args.table_identifier,
    )
    ranges = planned_ranges(table)
    columns = projected_columns(table, args.projection_count)
    endpoint = urlparse(required_env("S3_ENDPOINT"))
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        raise ValueError("S3_ENDPOINT must be an http(s) URL")
    filesystem = S3FileSystem(
        access_key=required_env("S3_ACCESS_KEY"),
        secret_key=required_env("S3_SECRET_KEY"),
        region=required_env("S3_REGION"),
        endpoint_override=endpoint.netloc,
        scheme=endpoint.scheme,
        force_virtual_addressing=False,
    )
    payload = {
        "table": args.table_identifier,
        "files": len(ranges),
        "rows": sum(item.rows for item in ranges),
        "projection": list(columns),
        "concurrency": args.concurrency,
        "iceberg_table_io": benchmark(
            "iceberg_table_io",
            lambda item, selected: read_via_table_io(table, item, selected),
            ranges,
            columns,
            args.repetitions,
            args.concurrency,
        ),
        "direct_s3": benchmark(
            "direct_s3",
            lambda item, selected: read_via_s3(filesystem, item, selected),
            ranges,
            columns,
            args.repetitions,
            args.concurrency,
        ),
    }
    write_json(args.output, payload)
    print(f"source_read_benchmark=true output={args.output}")


if __name__ == "__main__":
    main()
