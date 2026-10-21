from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    load_iceberg_table,
)
from flare.data.schema.feature_catalog import FeatureCatalog


def planned_data_files(table: Any) -> tuple[Any, ...]:
    files = []
    for task in table.scan().plan_files():
        if task.delete_files:
            raise NotImplementedError(
                "validation does not support Iceberg delete files"
            )
        files.append(task.file)
    return tuple(files)


def validate_schema(
    *,
    table_identifier: str,
    catalog_uri: str,
    manifest_dir: Path,
    warehouse: str | None = None,
) -> None:
    feature_catalog = FeatureCatalog.from_manifest_dir(manifest_dir)
    expected = {*feature_catalog.columns, "_ds", "_hour"}
    table = load_iceberg_table(
        build_iceberg_catalog_kwargs(
            name="gravitino-e2e",
            catalog_type="rest",
            uri=catalog_uri,
            warehouse=warehouse,
        ),
        table_identifier,
    )
    files = [str(data_file.file_path) for data_file in planned_data_files(table)]
    if not files:
        raise RuntimeError("Iceberg snapshot has no data files")

    for path in files:
        stream = table.io.new_input(path).open()
        try:
            parquet = pq.ParquetFile(stream)
            actual = set(parquet.schema_arrow.names)
            codecs = {
                parquet.metadata.row_group(row_group_index)
                .column(column_index)
                .compression
                for row_group_index in range(parquet.metadata.num_row_groups)
                for column_index in range(
                    parquet.metadata.row_group(row_group_index).num_columns
                )
            }
        finally:
            stream.close()
        if codecs != {"ZSTD"}:
            raise RuntimeError(
                f"Iceberg Parquet file must use only ZSTD compression: "
                f"path={path!r} codecs={sorted(codecs)!r}"
            )
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(
                "Iceberg Parquet schema differs from feature manifest: "
                f"path={path!r} missing={missing!r} unexpected={unexpected!r}"
            )
    print(
        f"physical_schema_files={len(files)} columns={len(expected)} compression=ZSTD"
    )


def validate_rows(
    *,
    table_identifier: str,
    catalog_uri: str,
    expected_rows: int,
    warehouse: str | None = None,
) -> None:
    table = load_iceberg_table(
        build_iceberg_catalog_kwargs(
            name="gravitino-e2e",
            catalog_type="rest",
            uri=catalog_uri,
            warehouse=warehouse,
        ),
        table_identifier,
    )
    files: dict[str, int] = {}
    for data_file in planned_data_files(table):
        path = str(data_file.file_path)
        if path in files:
            raise AssertionError(f"Iceberg scan planned a duplicate data file: {path}")
        files[path] = int(data_file.record_count)
    actual_rows = sum(files.values())
    if actual_rows != expected_rows:
        raise AssertionError(
            f"Iceberg row count differs: expected={expected_rows} actual={actual_rows}"
        )
    print(f"iceberg_rows={actual_rows} unique_data_files={len(files)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("table_identifier")
    schema_parser.add_argument("catalog_uri")
    schema_parser.add_argument("manifest_dir", type=Path)
    schema_parser.add_argument("--warehouse")

    rows_parser = subparsers.add_parser("rows")
    rows_parser.add_argument("table_identifier")
    rows_parser.add_argument("catalog_uri")
    rows_parser.add_argument("expected_rows", type=int)
    rows_parser.add_argument("--warehouse")

    args = parser.parse_args()

    if args.command == "schema":
        validate_schema(
            table_identifier=args.table_identifier,
            catalog_uri=args.catalog_uri,
            manifest_dir=args.manifest_dir,
            warehouse=args.warehouse,
        )
    elif args.command == "rows":
        validate_rows(
            table_identifier=args.table_identifier,
            catalog_uri=args.catalog_uri,
            expected_rows=args.expected_rows,
            warehouse=args.warehouse,
        )


if __name__ == "__main__":
    main()
