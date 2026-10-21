"""Iceberg catalog configuration and Gravitino connection settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

from flare.config.iceberg import IcebergSettings
from flare.io.aws import PYICEBERG_FSSPEC_FILE_IO

_REST_ACCESS_DELEGATION_HEADER = "header.X-Iceberg-Access-Delegation"
_REST_ACCESS_DELEGATION = "vended-credentials"
_REST_NOOP_AUTH = {"type": "noop"}


def build_iceberg_catalog_kwargs(
    *,
    name: str | None = None,
    catalog_type: str | None = None,
    uri: str | None = None,
    warehouse: str | None = None,
    extra: Mapping[str, Any] | None = None,
    settings: IcebergSettings | None = None,
) -> dict[str, Any]:
    if settings is not None:
        if (
            any(value is not None for value in (name, catalog_type, uri, warehouse))
            or extra
        ):
            raise ValueError(
                "settings cannot be combined with individual catalog arguments"
            )
        return normalize_iceberg_catalog_kwargs(settings.to_pyiceberg_kwargs())
    kwargs: dict[str, Any] = {}
    if extra:
        kwargs.update({key: value for key, value in extra.items() if value is not None})
    for key, value in (
        ("name", name),
        ("type", catalog_type),
        ("uri", uri),
        ("warehouse", warehouse),
    ):
        if value is not None:
            kwargs[key] = value
    return normalize_iceberg_catalog_kwargs(kwargs)


def normalize_iceberg_catalog_kwargs(
    catalog_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = {
        key: value for key, value in dict(catalog_kwargs).items() if value is not None
    }
    name = kwargs.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Iceberg catalog kwargs must include non-empty 'name'")

    catalog_type = kwargs.get("type")
    if not isinstance(catalog_type, str) or not catalog_type.strip():
        raise ValueError("Iceberg catalog kwargs must include non-empty 'type'")
    catalog_type = catalog_type.strip()

    if catalog_type == "gravitino_rest":
        catalog_type = "rest"
    kwargs["type"] = catalog_type

    if catalog_type == "rest":
        uri = kwargs.get("uri")
        if not uri:
            raise ValueError("REST Iceberg catalog requires non-empty 'uri'")
        parsed = urlparse(uri.rstrip("/"))
        if "/api/metalakes/" in parsed.path:
            kwargs["uri"] = urlunparse(
                (parsed.scheme, parsed.netloc, "/iceberg", "", "", "")
            )
        else:
            kwargs["uri"] = uri.rstrip("/")

        if kwargs["uri"].endswith("/iceberg") and not kwargs.get("warehouse"):
            raise ValueError(
                "Gravitino Iceberg REST catalog requires non-empty 'warehouse' specifying the catalog name"
            )

        auth = kwargs.get("auth")
        if auth is not None:
            if not isinstance(auth, dict) or auth.get("type") not in {
                "noop",
                "basic",
                "oauth2",
            }:
                raise ValueError(
                    "REST custom auth managers are unsupported; auth must be noop, basic, or oauth2"
                )
            if auth.get("type") == "basic" and not isinstance(auth.get("basic"), dict):
                raise ValueError("REST basic auth requires an auth.basic object")
            if auth.get("type") == "oauth2" and not isinstance(
                auth.get("oauth2"), dict
            ):
                raise ValueError("REST oauth2 auth requires an auth.oauth2 object")
        delegation = kwargs.get(_REST_ACCESS_DELEGATION_HEADER)
        if delegation is not None and delegation != _REST_ACCESS_DELEGATION:
            raise ValueError("REST access delegation must request vended-credentials")
        if "token" in kwargs and "header.Authorization" in kwargs:
            raise ValueError(
                "REST bearer authentication must use either token or header.Authorization"
            )
        if "header.Authorization" in kwargs:
            raise ValueError(
                "use token or auth configuration instead of header.Authorization"
            )
        if "token" not in kwargs and "auth" not in kwargs:
            kwargs["auth"] = dict(_REST_NOOP_AUTH)
        kwargs[_REST_ACCESS_DELEGATION_HEADER] = _REST_ACCESS_DELEGATION
        kwargs.setdefault("py-io-impl", PYICEBERG_FSSPEC_FILE_IO)
    if catalog_type == "in-memory" and not kwargs.get("warehouse"):
        raise ValueError("in-memory Iceberg catalog requires 'warehouse'")

    return kwargs


def load_iceberg_catalog(catalog_kwargs: Mapping[str, Any]):
    from pyiceberg.catalog import load_catalog

    kwargs = normalize_iceberg_catalog_kwargs(catalog_kwargs)
    catalog_name = kwargs.pop("name")
    return load_catalog(catalog_name, **kwargs)


def load_iceberg_table(catalog_kwargs: Mapping[str, Any], table_identifier: str):
    return load_iceberg_catalog(catalog_kwargs).load_table(table_identifier)
