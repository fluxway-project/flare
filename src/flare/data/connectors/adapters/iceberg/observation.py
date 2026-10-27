"""Read-only facts about one committed Iceberg publication snapshot."""

from __future__ import annotations

import time
from contextlib import closing
from dataclasses import dataclass, replace
from typing import Any

from flare.data.connectors.adapters.iceberg.access import (
    IcebergCatalogProvider,
    IcebergTableSession,
)
from flare.data.pipeline.production.publication.contract import (
    PUBLICATION_ID_PROPERTY,
    validate_publication_id,
)


@dataclass(frozen=True, slots=True)
class IcebergFileLayout:
    data_file_count: int
    data_file_bytes: int
    smallest_file_bytes: int
    p50_file_bytes: int
    p95_file_bytes: int
    largest_file_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.data_file_count,
            self.data_file_bytes,
            self.smallest_file_bytes,
            self.p50_file_bytes,
            self.p95_file_bytes,
            self.largest_file_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("Iceberg file layout values must not be negative")
        if self.data_file_count == 0 and any(values[1:]):
            raise ValueError("empty Iceberg file layout must have zero byte statistics")
        if self.data_file_count > 0 and self.smallest_file_bytes <= 0:
            raise ValueError("non-empty Iceberg file layout requires positive sizes")
        if not (
            self.smallest_file_bytes
            <= self.p50_file_bytes
            <= self.p95_file_bytes
            <= self.largest_file_bytes
        ):
            raise ValueError("Iceberg file layout size statistics are not ordered")


@dataclass(frozen=True, slots=True)
class IcebergCommitObservation:
    snapshot_id: int
    publication_id: str
    added_records: int
    write_seconds: float
    snapshot_observation_seconds: float
    file_layout: IcebergFileLayout | None

    def __post_init__(self) -> None:
        if self.snapshot_id <= 0:
            raise ValueError("Iceberg observation snapshot_id must be positive")
        validate_publication_id(self.publication_id)
        if self.added_records < 0:
            raise ValueError("Iceberg observation added_records must not be negative")
        if self.write_seconds < 0.0 or self.snapshot_observation_seconds < 0.0:
            raise ValueError("Iceberg observation durations must not be negative")


def observe_committed_publication(
    *,
    catalog_provider: IcebergCatalogProvider,
    table_identifier: str,
    expected_publication_id: str,
    write_seconds: float,
    include_file_layout: bool,
) -> IcebergCommitObservation:
    """Read and fence one committed publication without mutating Iceberg state."""
    started = time.perf_counter()
    with closing(IcebergTableSession(catalog_provider, table_identifier)) as session:
        observation = _observe_committed_snapshot(
            session.table(),
            expected_publication_id=expected_publication_id,
            write_seconds=write_seconds,
            include_file_layout=include_file_layout,
            snapshot_observation_seconds=0.0,
        )
    return replace(
        observation,
        snapshot_observation_seconds=time.perf_counter() - started,
    )


def _observe_committed_snapshot(
    table: Any,
    *,
    expected_publication_id: str,
    write_seconds: float,
    include_file_layout: bool,
    snapshot_observation_seconds: float,
) -> IcebergCommitObservation:
    publication_id = validate_publication_id(expected_publication_id)
    snapshot = table.current_snapshot()
    if snapshot is None:
        raise ValueError("Iceberg table has no snapshot after write")
    snapshot_id = getattr(snapshot, "snapshot_id", None)
    if (
        not isinstance(snapshot_id, int)
        or isinstance(snapshot_id, bool)
        or snapshot_id <= 0
    ):
        raise ValueError("Iceberg snapshot has an invalid snapshot id")
    summary = getattr(snapshot, "summary", None)
    if summary is None:
        raise ValueError("Iceberg snapshot has no summary after write")
    summary_get = getattr(summary, "get", None)
    if not callable(summary_get):
        raise TypeError("Iceberg snapshot summary must support key lookup")
    actual_publication_id = summary_get(PUBLICATION_ID_PROPERTY)
    if actual_publication_id != publication_id:
        raise RuntimeError(
            "Iceberg current snapshot publication id mismatch: "
            f"expected={publication_id!r} actual={actual_publication_id!r}"
        )
    added_records = summary_get("added-records")
    if added_records is None:
        raise ValueError("Iceberg snapshot summary missing 'added-records' after write")
    if isinstance(added_records, bool) or not isinstance(added_records, (str, int)):
        raise TypeError("Iceberg added-records must be an integer or integer string")
    added_records = int(added_records)
    if added_records < 0:
        raise ValueError("Iceberg snapshot added-records must not be negative")
    if include_file_layout:
        raise RuntimeError(
            "Iceberg file layout observation requires the immutable append index; "
            "full manifest observation is disabled"
        )
    file_layout = None
    return IcebergCommitObservation(
        snapshot_id=snapshot_id,
        publication_id=publication_id,
        added_records=added_records,
        write_seconds=write_seconds,
        snapshot_observation_seconds=snapshot_observation_seconds,
        file_layout=file_layout,
    )


__all__ = [
    "IcebergCommitObservation",
    "IcebergFileLayout",
    "observe_committed_publication",
]
