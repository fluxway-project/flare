from __future__ import annotations

from types import SimpleNamespace

import pytest

from flare.data.connectors.adapters.iceberg.observation import (
    _observe_committed_snapshot,
)


def _table(*, publication_id: str = "publication-1", entries=()):
    snapshot = SimpleNamespace(
        snapshot_id=17,
        summary={
            "flare.publication-id": publication_id,
            "added-records": "9",
        },
    )
    manifest = SimpleNamespace(
        fetch_manifest_entry=lambda file_io, discard_deleted: list(entries)
    )
    snapshot.manifests = lambda file_io: [manifest]
    return SimpleNamespace(current_snapshot=lambda: snapshot, io=object())


def _entry(size: int, *, snapshot_id: int = 17, status: str = "ADDED"):
    return SimpleNamespace(
        snapshot_id=snapshot_id,
        status=status,
        data_file=SimpleNamespace(file_size_in_bytes=size),
    )


def test_observation_fences_publication_and_keeps_layout_disabled_by_default() -> None:
    table = _table()
    table.current_snapshot().manifests = lambda _: (_ for _ in ()).throw(
        AssertionError("default observation must not read manifests")
    )
    observation = _observe_committed_snapshot(
        table,
        expected_publication_id="publication-1",
        write_seconds=1.5,
        include_file_layout=False,
        snapshot_observation_seconds=0.0,
    )

    assert observation.snapshot_id == 17
    assert observation.added_records == 9
    assert observation.file_layout is None


def test_observation_rejects_unbounded_file_layout_scan() -> None:
    with pytest.raises(RuntimeError, match="immutable append index"):
        _observe_committed_snapshot(
            _table(entries=[_entry(10), _entry(30), _entry(20)]),
            expected_publication_id="publication-1",
            write_seconds=0.0,
            include_file_layout=True,
            snapshot_observation_seconds=0.0,
        )


def test_observation_rejects_an_unfenced_current_snapshot() -> None:
    with pytest.raises(RuntimeError, match="publication id mismatch"):
        _observe_committed_snapshot(
            _table(publication_id="other"),
            expected_publication_id="publication-1",
            write_seconds=0.0,
            include_file_layout=False,
            snapshot_observation_seconds=0.0,
        )


def test_layout_requires_added_data_files() -> None:
    with pytest.raises(RuntimeError, match="immutable append index"):
        _observe_committed_snapshot(
            _table(entries=[_entry(10, status="EXISTING")]),
            expected_publication_id="publication-1",
            write_seconds=0.0,
            include_file_layout=True,
            snapshot_observation_seconds=0.0,
        )
