from __future__ import annotations

from typing import Any

import pytest

from flare.data.contracts.sink import SinkResult
from flare.data.pipeline.production.publication.contract import PublicationProtocol
from flare.data.pipeline.production.publication.manifest import PublicationManifest
from flare.data.pipeline.production.publication.session import execute_publication
from flare.data.pipeline.production.publication.window import (
    partition_publication_manifest,
)


class _Executor:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.executions: list[tuple[str, ...]] = []

    def execute(self, file_paths: tuple[str, ...], terminal: Any) -> SinkResult:
        self.executions.append(file_paths)
        if self.fails:
            raise RuntimeError("write failed")
        return SinkResult(total_rows=len(file_paths))


class _LeaseStore:
    def __init__(self) -> None:
        self.marked: list[tuple[str, list[str], PublicationProtocol]] = []
        self.completed: list[str] = []
        self.discarded: list[str] = []

    def mark_in_flight(
        self, publication_id: str, files: list[str], protocol: PublicationProtocol
    ) -> None:
        self.marked.append((publication_id, files, protocol))

    def complete_in_flight(self, publication_id: str) -> None:
        self.completed.append(publication_id)

    def discard_in_flight(self, publication_id: str) -> None:
        self.discarded.append(publication_id)


class _Terminal:
    def finalize(self, dataset: Any, *, schema: Any) -> SinkResult:
        raise AssertionError("manifest executor test must not finalize terminal")


def test_manifest_is_canonical_and_identity_is_order_independent() -> None:
    forward = PublicationManifest.from_files(["b", "a"])
    reverse = PublicationManifest.from_files(["a", "b"])

    assert forward.files == ("a", "b")
    assert forward.identity == reverse.identity


@pytest.mark.parametrize("files", [[], ["a", "a"], [""]])
def test_manifest_rejects_ambiguous_source_sets(files: list[str]) -> None:
    with pytest.raises(ValueError):
        PublicationManifest.from_files(files)


def test_manifest_windows_preserve_membership_and_identity_boundaries() -> None:
    manifest = PublicationManifest.from_files(["d", "b", "c", "a", "e"])

    windows = partition_publication_manifest(manifest, 2)

    assert [window.files for window in windows] == [("a", "b"), ("c", "d"), ("e",)]
    assert len({window.identity.value for window in windows}) == len(windows)
    assert tuple(path for window in windows for path in window.files) == manifest.files


def test_manifest_windows_reject_non_positive_size() -> None:
    manifest = PublicationManifest.from_files(["a"])

    with pytest.raises(ValueError, match="files_per_window"):
        partition_publication_manifest(manifest, 0)


def test_transactional_manifest_uses_one_lease_and_one_execution() -> None:
    manifest = PublicationManifest.from_files(["b", "a"])
    executor = _Executor()
    leases = _LeaseStore()

    result = execute_publication(
        manifest=manifest,
        executor=executor,
        terminal=_Terminal(),
        protocol=PublicationProtocol.ICEBERG_SNAPSHOT,
        lease_store=leases,
    )

    assert result.total_rows == 2
    assert executor.executions == [("a", "b")]
    assert leases.marked == [
        (manifest.identity.value, ["a", "b"], PublicationProtocol.ICEBERG_SNAPSHOT)
    ]
    assert leases.completed == [manifest.identity.value]
    assert leases.discarded == []


def test_transactional_failure_retains_lease_for_snapshot_reconciliation() -> None:
    manifest = PublicationManifest.from_files(["a"])
    leases = _LeaseStore()

    with pytest.raises(RuntimeError, match="write failed"):
        execute_publication(
            manifest=manifest,
            executor=_Executor(fails=True),
            terminal=_Terminal(),
            protocol=PublicationProtocol.ICEBERG_SNAPSHOT,
            lease_store=leases,
        )

    assert leases.completed == []
    assert leases.discarded == []


def test_non_transactional_failure_discards_lease() -> None:
    manifest = PublicationManifest.from_files(["a"])
    leases = _LeaseStore()

    with pytest.raises(RuntimeError, match="write failed"):
        execute_publication(
            manifest=manifest,
            executor=_Executor(fails=True),
            terminal=_Terminal(),
            protocol=PublicationProtocol.NON_TRANSACTIONAL,
            lease_store=leases,
        )

    assert leases.discarded == [manifest.identity.value]
