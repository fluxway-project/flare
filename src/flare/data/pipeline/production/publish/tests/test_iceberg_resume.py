from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flare.data.connectors.adapters.iceberg.recovery import reconcile_iceberg_in_flight
from flare.data.pipeline.production.publication.contract import (
    PublicationLease,
    PublicationProtocol,
)


def _lease(protocol=PublicationProtocol.ICEBERG_SNAPSHOT):
    return PublicationLease(files=("a", "b"), protocol=protocol)


def _snapshot(summary):
    return SimpleNamespace(summary=summary)


def _reconcile(snapshots, in_flight=None):
    state_manager = Mock()
    table = Mock()
    table.snapshots.return_value = snapshots
    with patch(
        "flare.data.connectors.adapters.iceberg.recovery.load_iceberg_table",
        return_value=table,
    ):
        reconcile_iceberg_in_flight(
            state_manager,
            in_flight or {"abc": _lease()},
            catalog_kwargs={"name": "catalog"},
            table_identifier="db.table",
        )
    return state_manager


def test_committed_publication_completes_lease() -> None:
    state_manager = _reconcile([_snapshot({"flare.publication-id": "abc"})])

    state_manager.complete_in_flight.assert_called_once_with("abc")
    state_manager.discard_in_flight.assert_not_called()


def test_uncommitted_publication_discards_lease_for_retry() -> None:
    state_manager = _reconcile([_snapshot({"flare.publication-id": "previous"})])

    state_manager.discard_in_flight.assert_called_once_with("abc")
    state_manager.complete_in_flight.assert_not_called()


def test_empty_table_discards_lease_for_retry() -> None:
    state_manager = _reconcile([])

    state_manager.discard_in_flight.assert_called_once_with("abc")


def test_duplicate_publication_id_is_rejected() -> None:
    snapshots = [
        _snapshot({"flare.publication-id": "abc"}),
        _snapshot({"flare.publication-id": "abc"}),
    ]

    with pytest.raises(RuntimeError, match="duplicate publication ids: abc"):
        _reconcile(snapshots)


@pytest.mark.parametrize("summary", [None, []])
def test_invalid_snapshot_summary_is_rejected(summary) -> None:
    with pytest.raises(RuntimeError, match="snapshot summary"):
        _reconcile([_snapshot(summary)])


def test_snapshot_without_publication_id_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="missing publication id"):
        _reconcile([_snapshot({"operation": "append"})])


def test_non_iceberg_lease_is_rejected_before_catalog_load() -> None:
    with pytest.raises(RuntimeError, match="non-Iceberg publications: abc"):
        _reconcile(
            [],
            {"abc": _lease(PublicationProtocol.NON_TRANSACTIONAL)},
        )


def test_commit_crash_window_is_completed_without_reexecution() -> None:
    pipeline = Mock()
    state_manager = _reconcile([_snapshot({"flare.publication-id": "abc"})])

    pipeline.execute.assert_not_called()
    state_manager.complete_in_flight.assert_called_once_with("abc")
