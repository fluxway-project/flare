"""Create an isolated Gravitino-managed Iceberg catalog."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

SECRET_ENVIRONMENT = (
    "GRAVITINO_JDBC_USER",
    "GRAVITINO_JDBC_PASSWORD",
    "GRAVITINO_S3_ACCESS_KEY",
    "GRAVITINO_S3_SECRET_KEY",
)
TOKEN_ENVIRONMENT = "GRAVITINO_TOKEN"


@dataclass(frozen=True)
class CatalogDefinition:
    api_uri: str
    metalake: str
    catalog_name: str
    backend_name: str
    warehouse: str
    jdbc_uri: str
    jdbc_driver: str
    s3_endpoint: str
    s3_region: str
    s3_path_style_access: bool
    s3_chunked_encoding_enabled: bool

    @property
    def collection_url(self) -> str:
        return f"{self.api_uri.rstrip('/')}/api/metalakes/{self.metalake}/catalogs"


def require_secrets(environment: Mapping[str, str]) -> dict[str, str]:
    missing = [name for name in SECRET_ENVIRONMENT if not environment.get(name)]
    if missing:
        raise ValueError(
            "missing credential environment variables: " + ", ".join(missing)
        )
    return {name: environment[name] for name in SECRET_ENVIRONMENT}


def catalog_payload(
    definition: CatalogDefinition, secrets: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "name": definition.catalog_name,
        "type": "RELATIONAL",
        "provider": "lakehouse-iceberg",
        "comment": "Isolated Iceberg catalog for streaming validation",
        "properties": {
            "catalog-backend": "jdbc",
            "catalog-backend-name": definition.backend_name,
            "credential-providers": "s3-secret-key",
            "uri": definition.jdbc_uri,
            "jdbc-driver": definition.jdbc_driver,
            "jdbc-user": secrets["GRAVITINO_JDBC_USER"],
            "jdbc-password": secrets["GRAVITINO_JDBC_PASSWORD"],
            "jdbc-initialize": "true",
            "warehouse": definition.warehouse,
            "io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
            "s3-endpoint": definition.s3_endpoint,
            "s3-region": definition.s3_region,
            "s3-access-key-id": secrets["GRAVITINO_S3_ACCESS_KEY"],
            "s3-secret-access-key": secrets["GRAVITINO_S3_SECRET_KEY"],
            "s3-path-style-access": str(definition.s3_path_style_access).lower(),
            "gravitino.bypass.s3.chunked-encoding-enabled": str(
                definition.s3_chunked_encoding_enabled
            ).lower(),
        },
    }


def require_token(environment: Mapping[str, str]) -> str:
    token = environment.get(TOKEN_ENVIRONMENT)
    if not token:
        raise ValueError(
            f"missing credential environment variable: {TOKEN_ENVIRONMENT}"
        )
    return token


def create_catalog(
    definition: CatalogDefinition, secrets: Mapping[str, str], token: str
) -> dict[str, Any]:
    response = requests.post(
        definition.collection_url,
        headers={
            "Accept": "application/vnd.gravitino.v1+json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=catalog_payload(definition, secrets),
        timeout=60,
    )
    if not response.ok:
        try:
            error = response.json()
        except ValueError:
            error = {"response_text": response.text[:1_000]}
        raise RuntimeError(
            json.dumps(
                {
                    "catalog_creation_http_status": response.status_code,
                    "catalog_creation_error": error,
                },
                sort_keys=True,
            )
        )
    response.raise_for_status()
    catalog = response.json().get("catalog")
    if not isinstance(catalog, dict):
        raise TypeError("Gravitino response does not contain a catalog object")
    if catalog.get("name") != definition.catalog_name:
        raise RuntimeError(f"unexpected catalog name: {catalog.get('name')!r}")
    return catalog


def parse_bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("value must be true or false")
    return value == "true"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-uri", required=True)
    parser.add_argument("--metalake", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--backend-name", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--jdbc-uri", required=True)
    parser.add_argument("--jdbc-driver", required=True)
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-region", required=True)
    parser.add_argument("--s3-path-style-access", required=True, type=parse_bool)
    parser.add_argument("--s3-chunked-encoding-enabled", required=True, type=parse_bool)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    definition = CatalogDefinition(
        api_uri=args.api_uri,
        metalake=args.metalake,
        catalog_name=args.catalog,
        backend_name=args.backend_name,
        warehouse=args.warehouse,
        jdbc_uri=args.jdbc_uri,
        jdbc_driver=args.jdbc_driver,
        s3_endpoint=args.s3_endpoint,
        s3_region=args.s3_region,
        s3_path_style_access=args.s3_path_style_access,
        s3_chunked_encoding_enabled=args.s3_chunked_encoding_enabled,
    )
    catalog = create_catalog(
        definition,
        require_secrets(os.environ),
        require_token(os.environ),
    )
    print(
        json.dumps(
            {
                "catalog_created": True,
                "metalake": definition.metalake,
                "catalog": catalog["name"],
                "provider": catalog.get("provider"),
                "warehouse": catalog.get("properties", {}).get("warehouse"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
