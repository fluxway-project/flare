from __future__ import annotations

import argparse

from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    load_iceberg_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_identifier")
    parser.add_argument("--rest-uri", required=True)
    parser.add_argument("--warehouse")
    parser.add_argument("--allowed-namespace", required=True)
    parser.add_argument("--drop-namespace", action="store_true")
    parser.add_argument("--purge", action="store_true")
    return parser.parse_args()


def cleanup_registered_table(
    *,
    table_identifier: str,
    catalog,
    purge: bool,
) -> bool:
    if not catalog.table_exists(table_identifier):
        return False
    if purge:
        catalog.purge_table(table_identifier)
    else:
        catalog.drop_table(table_identifier)
    if catalog.table_exists(table_identifier):
        raise AssertionError("table remains registered after drop")
    return True


def main() -> None:
    args = parse_args()
    namespace, separator, table_name = args.table_identifier.partition(".")
    if separator != "." or namespace != args.allowed_namespace:
        raise ValueError("cleanup table namespace differs from --allowed-namespace")
    if not table_name.startswith("manifest_e2e_"):
        raise ValueError("cleanup only accepts manifest_e2e tables")

    catalog = load_iceberg_catalog(
        build_iceberg_catalog_kwargs(
            name="gravitino-cleanup",
            catalog_type="rest",
            uri=args.rest_uri,
            warehouse=args.warehouse,
        )
    )
    cleaned = cleanup_registered_table(
        table_identifier=args.table_identifier,
        catalog=catalog,
        purge=args.purge,
    )
    if args.drop_namespace:
        catalog.drop_namespace(namespace)
    state = "removed" if cleaned else "absent"
    print(f"gravitino_e2e_table_cleanup={state}:{args.table_identifier}")


if __name__ == "__main__":
    main()
