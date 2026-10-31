"""Validated publication facts shared by Iceberg writing, recovery and reporting."""

from collections.abc import Iterable
from typing import Any

from flare.data.contracts.publication import (
    PUBLICATION_ID_PROPERTY,
    validate_publication_id,
)


def _snapshot_summary(snapshot: Any) -> Any:
    summary = getattr(snapshot, "summary", None)
    if summary is None:
        raise RuntimeError("Iceberg snapshot summary is missing")
    if not callable(getattr(summary, "get", None)):
        raise TypeError("Iceberg snapshot summary must support key lookup")
    return summary


def snapshot_publication_id(snapshot: Any, *, required: bool = True) -> str | None:
    publication_id = _snapshot_summary(snapshot).get(PUBLICATION_ID_PROPERTY)
    if publication_id is None:
        if not required:
            return None
        raise RuntimeError(
            "Iceberg snapshot summary is missing publication id: "
            f"snapshot_id={getattr(snapshot, 'snapshot_id', None)}"
        )
    try:
        return validate_publication_id(publication_id)
    except ValueError as exc:
        raise RuntimeError("Iceberg snapshot publication id is invalid") from exc


def index_publication_snapshots(
    snapshots: Iterable[Any], *, require_publication_ids: bool = True
) -> dict[str, Any]:
    """Index committed publications, rejecting ambiguous duplicate identities."""
    indexed = {}
    duplicated = set()
    for snapshot in snapshots:
        publication_id = snapshot_publication_id(
            snapshot, required=require_publication_ids
        )
        if publication_id is None:
            continue
        if publication_id in indexed:
            duplicated.add(publication_id)
        indexed[publication_id] = snapshot
    if duplicated:
        raise RuntimeError(
            "Iceberg contains duplicate publication ids: "
            + ", ".join(sorted(duplicated))
        )
    return indexed


def snapshot_added_records(snapshot: Any) -> int:
    value = _snapshot_summary(snapshot).get("added-records")
    if value is None:
        raise ValueError("Iceberg snapshot summary missing 'added-records'")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("Iceberg added-records must be an integer or integer string")
    records = int(value)
    if records < 0:
        raise ValueError("Iceberg snapshot added-records must not be negative")
    return records
