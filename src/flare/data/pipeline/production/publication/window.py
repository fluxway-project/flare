"""Deterministic bounded publication windows."""

from __future__ import annotations

from flare.data.pipeline.production.publication.manifest import PublicationManifest


def partition_publication_manifest(
    manifest: PublicationManifest,
    files_per_window: int,
) -> tuple[PublicationManifest, ...]:
    if files_per_window <= 0:
        raise ValueError(f"files_per_window must be positive, got {files_per_window}")
    windows = tuple(
        PublicationManifest.from_files(manifest.files[start : start + files_per_window])
        for start in range(0, len(manifest.files), files_per_window)
    )
    if tuple(path for window in windows for path in window.files) != manifest.files:
        raise RuntimeError("publication windowing changed manifest membership")
    return windows
