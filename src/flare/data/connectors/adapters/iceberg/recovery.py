"""Reconcile producer leases with committed Iceberg snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flare.data.connectors.adapters.iceberg.catalog import load_iceberg_table
from flare.data.pipeline.production.publication.contract import (
    PUBLICATION_ID_PROPERTY,
    PublicationLease,
    PublicationProtocol,
    validate_publication_id,
)


def reconcile_iceberg_in_flight(
    state_manager: Any,
    in_flight: Mapping[str, PublicationLease],
    *,
    catalog_kwargs: Mapping[str, Any],
    table_identifier: str,
) -> None:
    if not in_flight:
        return
    unsupported = sorted(
        publication_id
        for publication_id, lease in in_flight.items()
        if lease.protocol is not PublicationProtocol.ICEBERG_SNAPSHOT
    )
    if unsupported:
        raise RuntimeError(
            "Cannot reconcile non-Iceberg publications: " + ", ".join(unsupported)
        )
    table = load_iceberg_table(catalog_kwargs, table_identifier)
    snapshots = list(table.snapshots())

    committed_counts: dict[str, int] = {}
    for snapshot in snapshots:
        summary = getattr(snapshot, "summary", None)
        if summary is None:
            raise RuntimeError("Iceberg snapshot summary is missing")
        summary_get = getattr(summary, "get", None)
        if not callable(summary_get):
            raise RuntimeError(  # noqa: TRY004
                "Iceberg snapshot summary must support key lookup"
            )
        publication_id = summary_get(PUBLICATION_ID_PROPERTY)
        if publication_id is None:
            raise RuntimeError(
                "Iceberg snapshot summary is missing publication id: "
                f"snapshot_id={getattr(snapshot, 'snapshot_id', None)}"
            )
        try:
            publication_id = validate_publication_id(publication_id)
        except ValueError as exc:
            raise RuntimeError("Iceberg snapshot publication id is invalid") from exc
        committed_counts[publication_id] = committed_counts.get(publication_id, 0) + 1

    duplicated = sorted(
        publication_id
        for publication_id, count in committed_counts.items()
        if count > 1
    )
    if duplicated:
        raise RuntimeError(
            "Iceberg contains duplicate publication ids: " + ", ".join(duplicated)
        )

    for publication_id in in_flight:
        if publication_id in committed_counts:
            state_manager.complete_in_flight(publication_id)
        else:
            state_manager.discard_in_flight(publication_id)
