"""Run read-only preflight checks for the real Iceberg producer/consumer E2E."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    load_iceberg_table,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs-root", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--catalog-name", required=True)
    parser.add_argument("--catalog-type", required=True)
    parser.add_argument("--catalog-uri", required=True)
    parser.add_argument("--catalog-warehouse", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _input_inventory(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"HDFS root does not exist: {root}")
    partitions = sorted(root.glob("ds=*/hour=*"))
    if not partitions:
        raise FileNotFoundError(f"no ds/hour partitions found under {root}")
    digest = hashlib.sha256()
    file_count = 0
    first_file: Path | None = None
    last_file: Path | None = None
    for partition in partitions:
        if not partition.is_dir():
            continue
        for path in sorted(partition.glob("*.zst")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(root).as_posix()
            digest.update(relative_path.encode())
            digest.update(b"\n")
            file_count += 1
            if first_file is None:
                first_file = path
            last_file = path
    if first_file is None or last_file is None:
        raise FileNotFoundError(f"no .zst input files found under {root}")
    return {
        "root": str(root),
        "partition_count": len(partitions),
        "file_count": file_count,
        "first_file": str(first_file),
        "last_file": str(last_file),
        "relative_path_sha256": digest.hexdigest(),
    }


def _table_inventory(table: Any) -> dict[str, Any]:
    schema = table.schema()
    field_names = tuple(field.name for field in schema.fields)
    required_fields = ("_ds", "_hour")
    missing_fields = tuple(
        field_name for field_name in required_fields if field_name not in field_names
    )
    if missing_fields:
        raise ValueError(f"Iceberg table misses required fields: {missing_fields}")

    partition_spec = table.spec()
    partition_names = tuple(field.name for field in partition_spec.fields)
    if partition_names != required_fields:
        raise ValueError(
            "Iceberg table partition contract differs: "
            f"expected={required_fields!r} actual={partition_names!r}"
        )

    snapshot = table.current_snapshot()
    return {
        "properties": dict(table.properties),
        "schema_column_count": len(field_names),
        "schema_columns": list(field_names),
        "partition_fields": list(partition_names),
        "current_snapshot_id": None if snapshot is None else snapshot.snapshot_id,
        "current_snapshot_operation": (
            None if snapshot is None else snapshot.summary.operation.value
        ),
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    input_inventory = _input_inventory(args.hdfs_root)
    token = os.environ.get("GRAVITINO_TOKEN")
    if not token:
        raise RuntimeError("GRAVITINO_TOKEN must be injected at runtime")
    catalog_kwargs = build_iceberg_catalog_kwargs(
        name=args.catalog_name,
        catalog_type=args.catalog_type,
        uri=args.catalog_uri,
        warehouse=args.catalog_warehouse,
        extra={"token": token},
    )
    table = load_iceberg_table(catalog_kwargs, args.table)
    table_inventory = _table_inventory(table)
    return {
        "table": args.table,
        "catalog": {
            "name": args.catalog_name,
            "type": args.catalog_type,
            "uri": args.catalog_uri,
            "warehouse": args.catalog_warehouse,
        },
        "input": input_inventory,
        "iceberg": table_inventory,
        "write_performed": False,
        "checkpoint_accessed": False,
    }


def main() -> None:
    args = _parse_args()
    result = run_preflight(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
