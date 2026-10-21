"""Finite transactional publication coordination."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flare.data.execution.plan.terminal import DatasetTerminal
from flare.data.pipeline.production.publication.contract import (
    PublicationLeaseStore,
    PublicationProtocol,
    PublicationResult,
)
from flare.data.pipeline.production.publication.manifest import PublicationManifest


@runtime_checkable
class ManifestExecutor(Protocol):
    def execute(
        self,
        file_paths: tuple[str, ...],
        terminal: DatasetTerminal,
    ) -> PublicationResult: ...


def execute_publication(
    *,
    manifest: PublicationManifest,
    executor: ManifestExecutor,
    terminal: DatasetTerminal,
    protocol: PublicationProtocol,
    lease_store: PublicationLeaseStore,
) -> PublicationResult:
    """Execute exactly one finite source manifest under one publication lease."""
    publication_id = manifest.identity.value
    lease_store.mark_in_flight(publication_id, list(manifest.files), protocol)
    try:
        result = executor.execute(manifest.files, terminal)
    except Exception:
        if protocol is PublicationProtocol.NON_TRANSACTIONAL:
            lease_store.discard_in_flight(publication_id)
        raise
    lease_store.complete_in_flight(publication_id)
    return result
