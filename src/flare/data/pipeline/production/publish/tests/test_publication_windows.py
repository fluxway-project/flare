from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from flare.data.contracts.sink import SinkResult
from flare.data.pipeline.production.publication.contract import PublicationProtocol
from flare.data.pipeline.production.publish.runner import (
    discover_publication_manifest,
    publish_manifest,
)


@dataclass
class _Pipeline:
    executions: list[tuple[str, ...]]

    def execute(self, file_paths: tuple[str, ...], terminal: Any) -> SinkResult:
        self.executions.append(file_paths)
        return SinkResult(total_rows=len(file_paths))


class _StateManager:
    def __init__(self, pending: set[str]) -> None:
        self._pending = pending
        self.marked: list[tuple[str, list[str], PublicationProtocol]] = []
        self.completed: list[str] = []

    def is_pending(self, file_path: str) -> bool:
        return file_path in self._pending

    def mark_in_flight(
        self, publication_id: str, files: list[str], protocol: PublicationProtocol
    ) -> None:
        self.marked.append((publication_id, files, protocol))

    def complete_in_flight(self, publication_id: str) -> None:
        self.completed.append(publication_id)

    def discard_in_flight(self, publication_id: str) -> None:
        raise AssertionError("transactional publication must retain failed lease")


class _Reporter:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int, int]] = []

    def update(
        self, current: int, total: int, rows_processed: int, elapsed_batch: float
    ) -> None:
        self.updates.append((current, total, rows_processed))


class _Output:
    publication_protocol = PublicationProtocol.ICEBERG_SNAPSHOT

    def __init__(self) -> None:
        self.identities: list[str] = []

    def bind(self, identity: Any) -> _Output:
        self.identities.append(identity.value)
        return self


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        start_date="2026-01-01",
        end_date="2026-01-01",
        hdfs_root="/hdfs",
        limit_files=None,
        publication_window_files=2,
    )


def test_finite_hdfs_execution_uses_deterministic_windows_and_leases() -> None:
    paths = ["/hdfs/c.zst", "/hdfs/a.zst", "/hdfs/b.zst"]
    state = _StateManager(set(paths))
    pipeline = _Pipeline(executions=[])
    output = _Output()
    reporter = _Reporter()

    with patch(
        "flare.data.pipeline.production.publish.runner.iter_hdfs_file_paths",
        return_value=iter(paths),
    ):
        args = _args()
        args.input = None
        manifest = discover_publication_manifest(args, state)
        publish_manifest(
            manifest,
            args,
            state,
            pipeline,
            output,
            reporter,  # pyrefly: ignore[bad-argument-type]
        )

    assert pipeline.executions == [("/hdfs/a.zst", "/hdfs/b.zst"), ("/hdfs/c.zst",)]
    assert [leased_paths for _, leased_paths, _ in state.marked] == [
        ["/hdfs/a.zst", "/hdfs/b.zst"],
        ["/hdfs/c.zst"],
    ]
    assert all(
        protocol is PublicationProtocol.ICEBERG_SNAPSHOT
        for _, _, protocol in state.marked
    )
    publication_ids = [publication_id for publication_id, _, _ in state.marked]
    assert output.identities == publication_ids
    assert state.completed == publication_ids
    assert reporter.updates == [(2, 3, 2), (3, 3, 1)]
