"""Canonical digest operations for Iceberg source identities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("SHA-256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: Any, *, sort_keys: bool = True) -> str:
    encoded = json.dumps(
        value,
        sort_keys=sort_keys,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_sha256_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    if value.lower() != value:
        raise ValueError(f"{name} must use lowercase SHA-256 hex")


def normalize_table_uuid(value: object) -> str:
    if value is None:
        raise ValueError("Iceberg table metadata must provide a table UUID")
    normalized = str(value)
    if not normalized:
        raise ValueError("Iceberg table metadata must provide a table UUID")
    return normalized


def require_table_uuid(table: Any) -> str:
    metadata = getattr(table, "metadata", None)
    table_uuid = getattr(metadata, "table_uuid", None)
    if table_uuid is None:
        raise ValueError("Iceberg table metadata has no table UUID")
    if not str(table_uuid):
        raise ValueError("Iceberg table metadata has an empty table UUID")
    return str(table_uuid)


def bound_source_identity(base_identity: str, table_uuid: object) -> str:
    return f"{base_identity}&table_uuid={normalize_table_uuid(table_uuid)}"


def validate_table_uuid(table: Any, expected_table_uuid: str | None) -> None:
    metadata = getattr(table, "metadata", None)
    table_uuid = getattr(metadata, "table_uuid", None)
    if table_uuid is None or str(table_uuid) != expected_table_uuid:
        raise ValueError("Iceberg table UUID changed during source lifetime")
