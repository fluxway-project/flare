from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa

from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    load_iceberg_catalog,
)
from flare.data.connectors.adapters.iceberg.table import create_iceberg_table
from flare.data.pipeline.production.builder.row_parser import build_feature_processor
from flare.data.pipeline.production.publish.partitioning import (
    append_partition_fields,
)


def canonical_training_schema(conf_dir: Path, model_conf_dir: Path) -> pa.Schema:
    processor = build_feature_processor(conf_dir, model_conf_dir)
    return append_partition_fields(processor.build_vortex_schema())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_identifier")
    parser.add_argument("--rest-uri", required=True)
    parser.add_argument("--warehouse")
    parser.add_argument("--conf-dir", type=Path, required=True)
    parser.add_argument("--model-conf-dir", type=Path, required=True)
    parser.add_argument("--writer-target-size-bytes", type=int, required=True)
    parser.add_argument("--create-namespace", action="store_true")
    args = parser.parse_args()

    namespace, separator, _ = args.table_identifier.rpartition(".")
    if separator != "." or not namespace:
        raise ValueError("table_identifier must include a namespace")
    catalog = load_iceberg_catalog(
        build_iceberg_catalog_kwargs(
            name="gravitino-e2e-provision",
            catalog_type="rest",
            uri=args.rest_uri,
            warehouse=args.warehouse,
        )
    )
    if args.create_namespace:
        catalog.create_namespace(namespace)
    schema = canonical_training_schema(args.conf_dir, args.model_conf_dir)
    create_iceberg_table(
        catalog=catalog,
        table_identifier=args.table_identifier,
        schema=schema,
        writer_target_size_bytes=args.writer_target_size_bytes,
    )
    print(
        f"iceberg_table_provisioned={args.table_identifier} "
        f"columns={len(schema)} partitions=_ds,_hour"
    )


if __name__ == "__main__":
    main()
