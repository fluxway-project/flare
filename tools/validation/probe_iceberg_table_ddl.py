"""Probe a Gravitino-managed Iceberg catalog without importing flare-jax."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from typing import Any

import requests
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType


@dataclass(frozen=True)
class ProbeTarget:
    gravitino_api_uri: str
    metalake: str
    catalog_name: str
    iceberg_rest_uri: str
    namespace: str

    @property
    def catalog_url(self) -> str:
        return (
            f"{self.gravitino_api_uri.rstrip('/')}/api/metalakes/"
            f"{self.metalake}/catalogs/{self.catalog_name}"
        )


def probe_schema() -> Schema:
    return Schema(
        NestedField(id=1, name="row_id", type=LongType(), required=True),  # pyrefly: ignore[missing-argument]
        NestedField(id=2, name="_ds", type=StringType(), required=False),  # pyrefly: ignore[missing-argument]
        NestedField(id=3, name="_hour", type=StringType(), required=False),  # pyrefly: ignore[missing-argument]
    )


def validate_probe_identifier(identifier: str) -> tuple[str, str]:
    namespace, separator, table = identifier.partition(".")
    if separator != "." or not namespace or not table:
        raise ValueError("probe table identifier must be namespace.table")
    if not table.startswith("manifest_e2e_ddl_probe_"):
        raise ValueError("probe table name must start with manifest_e2e_ddl_probe_")
    return namespace, table


def inspect_gravitino_catalog(target: ProbeTarget) -> dict[str, Any]:
    response = requests.get(
        target.catalog_url,
        headers={"Accept": "application/vnd.gravitino.v1+json"},
        timeout=30,
    )
    if response.status_code == 404:
        raise RuntimeError(f"Gravitino catalog not found: {target.catalog_url}")
    response.raise_for_status()
    catalog = response.json().get("catalog")
    if not isinstance(catalog, dict):
        raise TypeError("Gravitino response does not contain a catalog object")
    if catalog.get("name") != target.catalog_name:
        raise RuntimeError(f"unexpected catalog name: {catalog.get('name')!r}")
    return catalog


def create_and_purge_table(target: ProbeTarget, identifier: str) -> None:
    validate_probe_identifier(identifier)
    catalog = load_catalog(
        "standalone-gravitino-iceberg-ddl-probe",
        type="rest",
        uri=target.iceberg_rest_uri.rstrip("/"),
        warehouse=target.catalog_name,
        auth={"type": "noop"},  # pyrefly: ignore[bad-argument-type]
        **{"header.X-Iceberg-Access-Delegation": "vended-credentials"},
    )
    if catalog.table_exists(identifier):
        raise FileExistsError(identifier)
    catalog.create_table(identifier, schema=probe_schema())
    if not catalog.table_exists(identifier):
        raise AssertionError(f"table was not registered: {identifier}")
    catalog.purge_table(identifier)
    if catalog.table_exists(identifier):
        raise AssertionError(f"table was not purged: {identifier}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravitino-api-uri", required=True)
    parser.add_argument("--metalake", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--iceberg-rest-uri", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--table")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = ProbeTarget(
        gravitino_api_uri=args.gravitino_api_uri,
        metalake=args.metalake,
        catalog_name=args.catalog,
        iceberg_rest_uri=args.iceberg_rest_uri,
        namespace=args.namespace,
    )
    table = args.table or f"manifest_e2e_ddl_probe_{uuid.uuid4().hex}"
    identifier = f"{target.namespace}.{table}"
    validate_probe_identifier(identifier)
    catalog = inspect_gravitino_catalog(target)
    print(
        json.dumps(
            {
                "gravitino_catalog_visible": True,
                "metalake": target.metalake,
                "catalog": catalog.get("name"),
                "provider": catalog.get("provider"),
                "iceberg_rest_uri": target.iceberg_rest_uri.rstrip("/"),
                "table": identifier,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        create_and_purge_table(target, identifier)
    except Exception as exc:
        print(
            f"iceberg_rest_table_ddl=false table={identifier} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
    print(f"iceberg_rest_table_ddl=true table={identifier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
