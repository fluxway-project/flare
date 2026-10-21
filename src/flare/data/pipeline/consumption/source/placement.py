"""Deterministic placement-time binding for source partitions and files."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _validate_node(node_ordinal: int, node_count: int) -> None:
    if (
        not isinstance(node_count, int)
        or isinstance(node_count, bool)
        or node_count <= 0
    ):
        raise ValueError("node_count must be a positive integer")
    if (
        not isinstance(node_ordinal, int)
        or isinstance(node_ordinal, bool)
        or not 0 <= node_ordinal < node_count
    ):
        raise ValueError("node_ordinal must be within node_count")


def stable_owner(identity: str, node_count: int) -> int:
    if not isinstance(identity, str) or not identity:
        raise ValueError("partition identity must be a non-empty string")
    if (
        not isinstance(node_count, int)
        or isinstance(node_count, bool)
        or node_count <= 0
    ):
        raise ValueError("node_count must be a positive integer")
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % node_count


def bind_partitions(
    identities: Iterable[str], *, node_ordinal: int, node_count: int
) -> tuple[str, ...]:
    _validate_node(node_ordinal, node_count)
    normalized = tuple(identities)
    if any(not isinstance(identity, str) or not identity for identity in normalized):
        raise ValueError("partition identities must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("partition identities must be unique")
    return tuple(
        identity
        for identity in normalized
        if stable_owner(identity, node_count) == node_ordinal
    )


__all__ = ["bind_partitions", "stable_owner"]
