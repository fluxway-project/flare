"""Benchmark PyIceberg append commits against direct S3 Parquet writes."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    load_iceberg_catalog,
)
from flare.data.connectors.adapters.iceberg.table import iceberg_table_properties
from pyarrow.fs import S3FileSystem

from tools.validation.performance_metrics import Measurement, summarize, write_json


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required for direct S3 writes")
    return value


def make_data(rows: int, offset: int) -> pa.Table:
    if rows <= 0:
        raise ValueError("rows must be positive")
    return pa.table(
        {
            "row_id": pa.array(range(offset, offset + rows), type=pa.int64()),
            "value": pa.array([1.0] * rows, type=pa.float32()),
            "_ds": pa.array(["2026-07-23"] * rows),
            "_hour": pa.array(["00"] * rows),
        }
    )


def make_s3_files(
    filesystem, root: str, data: pa.Table, blocks: int, concurrency: int
) -> None:
    if blocks < 1:
        raise ValueError("blocks must be positive")
    batches = data.to_batches(max_chunksize=(data.num_rows + blocks - 1) // blocks)
    paths = [
        f"{root.rstrip('/')}/part-{index:05d}.parquet" for index in range(len(batches))
    ]

    def write_one(item) -> None:
        path, batch = item
        with filesystem.open_output_stream(path) as stream:
            pq.write_table(pa.Table.from_batches([batch]), stream, compression="zstd")

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    with ThreadPoolExecutor(max_workers=min(blocks, concurrency)) as executor:
        tuple(executor.map(write_one, zip(paths, batches, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_identifier")
    parser.add_argument("--rest-uri", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.concurrency < 1:
        raise ValueError("repetitions and concurrency must be positive")
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
    data = make_data(args.rows, 0)
    catalog = load_iceberg_catalog(
        build_iceberg_catalog_kwargs(
            name="publish-write-benchmark",
            catalog_type="rest",
            uri=args.rest_uri,
            warehouse=args.warehouse,
        )
    )
    table = catalog.load_table(args.table_identifier)
    properties = dict(table.properties)
    expected = iceberg_table_properties(int(properties["write.target-file-size-bytes"]))
    if any(properties.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "benchmark table properties do not match the Iceberg writer contract"
        )
    iceberg_measurements = []
    direct_measurements = []
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        table.append(make_data(args.rows, repetition * args.rows))
        iceberg_measurements.append(
            Measurement(time.perf_counter() - started, args.rows)
        )
        started = time.perf_counter()
        make_s3_files(
            filesystem,
            f"{args.s3_root.rstrip('/')}/run-{repetition}",
            data,
            args.blocks,
            args.concurrency,
        )
        direct_measurements.append(
            Measurement(time.perf_counter() - started, args.rows)
        )
    payload = {
        "table": args.table_identifier,
        "rows_per_repetition": args.rows,
        "blocks": args.blocks,
        "concurrency": args.concurrency,
        "iceberg_append": summarize(iceberg_measurements),
        "direct_s3_parquet": summarize(direct_measurements),
    }
    write_json(args.output, payload)
    print(f"publish_write_benchmark=true output={args.output}")


if __name__ == "__main__":
    main()
