#!/usr/bin/env python3
"""Minimal Gravitino REST + S3 + PyIceberg Parquet smoke test.

Dependencies: pyiceberg, pyarrow, requests.
Credentials are read from environment variables and are never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from pyarrow.fs import S3FileSystem
from pyiceberg.catalog import load_catalog

from flare.data.connectors.adapters.iceberg.catalog import build_iceberg_catalog_kwargs


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"missing required value: {name}")
    return value


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rest-uri", required=True, help="Gravitino Iceberg REST endpoint"
    )
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--table-location-root", required=True)
    parser.add_argument(
        "--keep-table",
        action="store_true",
        help="Keep the generated namespace/table for manual inspection",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class S3Settings:
    endpoint: str
    access_key: str
    secret_key: str
    region: str

    @classmethod
    def from_environment(cls) -> S3Settings:
        return cls(
            endpoint=_required(os.environ.get("S3_ENDPOINT"), "S3_ENDPOINT"),
            access_key=_required(os.environ.get("S3_ACCESS_KEY"), "S3_ACCESS_KEY"),
            secret_key=_required(os.environ.get("S3_SECRET_KEY"), "S3_SECRET_KEY"),
            region=_required(os.environ.get("S3_REGION"), "S3_REGION"),
        )


def _configure_s3(settings: S3Settings) -> None:
    os.environ["AWS_ENDPOINT_URL"] = settings.endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = settings.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.secret_key
    os.environ["AWS_REGION"] = settings.region
    os.environ["AWS_DEFAULT_REGION"] = settings.region
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    if settings.endpoint.startswith("http://"):
        os.environ["AWS_ALLOW_HTTP"] = "true"


def _check_health(url: str) -> None:
    response = requests.get(url, timeout=15)
    response.raise_for_status()


def _load_rest_catalog(args: argparse.Namespace):
    properties = build_iceberg_catalog_kwargs(
        name="gravitino-smoke",
        catalog_type="rest",
        uri=_required(args.rest_uri, "--rest-uri"),
        warehouse=args.warehouse,
    )
    catalog_name = properties.pop("name")
    return load_catalog(catalog_name, **properties)


def _snapshot_id(table) -> int:
    snapshot = table.current_snapshot()
    if snapshot is None:
        raise AssertionError("table has no current snapshot")
    return int(snapshot.snapshot_id)


def _file_paths(table, snapshot_id: int) -> set[str]:
    paths: set[str] = set()
    for task in table.scan(snapshot_id=snapshot_id).plan_files():
        data_file = getattr(task, "file", None) or getattr(task, "data_file", None)
        path = getattr(data_file, "file_path", None)
        if not path:
            raise AssertionError(f"Iceberg plan task has no file path: {task!r}")
        paths.add(str(path))
    return paths


def _read_s3_parquet(paths: set[str], settings: S3Settings) -> pa.Table:
    if not paths:
        raise AssertionError("append snapshot did not contain a new data file")
    parsed = [urlparse(path) for path in sorted(paths)]
    if any(item.scheme != "s3" or not item.netloc or not item.path for item in parsed):
        raise AssertionError(f"append files are not S3 paths: {sorted(paths)!r}")
    bucket = parsed[0].netloc
    if any(item.netloc != bucket for item in parsed):
        raise AssertionError("append files span multiple S3 buckets")
    fs = S3FileSystem(
        access_key=settings.access_key,
        secret_key=settings.secret_key,
        region=settings.region,
        endpoint_override=settings.endpoint,
    )
    tables = [pq.read_table(f"{bucket}{item.path}", filesystem=fs) for item in parsed]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _table_location(root: str, namespace: str) -> str:
    parsed = urlparse(root)
    prefix = parsed.path.strip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not prefix:
        raise ValueError("--table-location-root must be an S3 prefix below a bucket")
    return f"s3://{parsed.netloc}/{prefix}/{namespace}/parquet_probe"


def _delete_s3_location(location: str, settings: S3Settings) -> None:
    parsed = urlparse(location)
    fs = S3FileSystem(
        access_key=settings.access_key,
        secret_key=settings.secret_key,
        region=settings.region,
        endpoint_override=settings.endpoint,
    )
    fs.delete_dir(f"{parsed.netloc}{parsed.path}")


def main() -> int:
    args = _args()
    settings = S3Settings.from_environment()
    _configure_s3(settings)
    _check_health(args.health_url)

    catalog = _load_rest_catalog(args)
    namespace = f"smoke_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    identifier = f"{namespace}.parquet_probe"
    table_location = _table_location(args.table_location_root, namespace)
    namespace_created = False
    table_created = False
    try:
        catalog.create_namespace(namespace)
        namespace_created = True
        table_schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("value", pa.string(), nullable=False),
            ]
        )
        table = catalog.create_table(
            identifier,
            schema=table_schema,
            location=table_location,
            properties={"write.format.default": "parquet"},
        )
        table_created = True
        expected = pa.Table.from_arrays(
            [
                pa.array([0, 1, 2, 3], type=pa.int64()),
                pa.array(["zero", "one", "two", "three"]),
            ],
            schema=table_schema,
        )
        table.append(expected, snapshot_properties={"smoke-test": "true"})

        reopened = catalog.load_table(identifier)
        first_snapshot = _snapshot_id(reopened)
        first_files = _file_paths(reopened, first_snapshot)
        actual = reopened.scan().to_arrow()
        if actual.num_rows != expected.num_rows:
            raise AssertionError(
                f"row count mismatch: expected {expected.num_rows}, got {actual.num_rows}"
            )
        for column in expected.column_names:
            if actual[column].to_pylist() != expected[column].to_pylist():
                raise AssertionError(f"column mismatch: {column}")
        if reopened.current_snapshot() is None:
            raise AssertionError("append did not create an Iceberg snapshot")

        appended = pa.Table.from_arrays(
            [
                pa.array([4, 5, 6, 7], type=pa.int64()),
                pa.array(["four", "five", "six", "seven"]),
            ],
            schema=table_schema,
        )
        table.append(appended, snapshot_properties={"smoke-test": "append"})
        deadline = time.monotonic() + 30
        latest = reopened
        while _snapshot_id(latest) == first_snapshot and time.monotonic() < deadline:
            time.sleep(0.5)
            latest = catalog.load_table(identifier)
        second_snapshot = _snapshot_id(latest)
        if second_snapshot == first_snapshot:
            raise TimeoutError("REST catalog did not expose the appended snapshot")
        appended_files = _file_paths(latest, second_snapshot) - first_files
        streamed = _read_s3_parquet(appended_files, settings)
        if streamed.num_rows != appended.num_rows:
            raise AssertionError(
                f"streamed append row count mismatch: expected {appended.num_rows}, got {streamed.num_rows}"
            )
        for column in appended.column_names:
            if streamed[column].to_pylist() != appended[column].to_pylist():
                raise AssertionError(f"streamed append column mismatch: {column}")
        print("gravitino_rest=true")
        print("s3_parquet_write=true")
        print("s3_parquet_read=true")
        print("streaming_append_discovery=true")
        print("streaming_append_read=true")
        print(f"append_rows={streamed.num_rows}")
        print(f"rows={actual.num_rows}")
        print(f"table={identifier}")
        return 0
    finally:
        if not args.keep_table and table_created:
            catalog.drop_table(identifier)
        if not args.keep_table and namespace_created:
            catalog.drop_namespace(namespace)
        if not args.keep_table:
            _delete_s3_location(table_location, settings)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"smoke_test_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        raise
