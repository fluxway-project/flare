"""Reconcile producer leases with committed Iceberg snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from typing import Any

from flare.data.connectors.adapters.iceberg.access import (
    IcebergCatalogProvider,
    IcebergTableSession,
)
from flare.data.connectors.adapters.iceberg.publication import (
    index_publication_snapshots,
)
from flare.data.contracts.publication import (
    PublicationLease,
    PublicationProtocol,
)


def reconcile_iceberg_in_flight(
    state_manager: Any,
    in_flight: Mapping[str, PublicationLease],
    *,
    catalog_provider: IcebergCatalogProvider,
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
    with closing(IcebergTableSession(catalog_provider, table_identifier)) as session:
        committed = index_publication_snapshots(session.table().snapshots())

    for publication_id in in_flight:
        if publication_id in committed:
            state_manager.complete_in_flight(publication_id)
        else:
            state_manager.discard_in_flight(publication_id)
