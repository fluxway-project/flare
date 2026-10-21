"""Canonical finite source manifests for transactional publication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from flare.data.pipeline.production.publication.contract import PublicationIdentity


@dataclass(frozen=True, slots=True)
class PublicationManifest:
    """The complete immutable source set for one transactional execution."""

    files: tuple[str, ...]
    identity: PublicationIdentity

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("publication manifest must not be empty")
        if tuple(sorted(self.files)) != self.files:
            raise ValueError("publication manifest files must be sorted")
        if any(not path for path in self.files):
            raise ValueError("publication manifest files must be non-empty strings")
        if any(
            self.files[index] == self.files[index - 1]
            for index in range(1, len(self.files))
        ):
            raise ValueError("publication manifest files must be unique")

    @classmethod
    def from_files(cls, files: Iterable[str]) -> PublicationManifest:
        paths = tuple(files)
        if any(not isinstance(path, str) for path in paths):
            raise TypeError("publication manifest files must be strings")
        canonical_files = tuple(sorted(paths))
        canonical = json.dumps(
            canonical_files,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            files=canonical_files,
            identity=PublicationIdentity(hashlib.sha256(canonical).hexdigest()),
        )
