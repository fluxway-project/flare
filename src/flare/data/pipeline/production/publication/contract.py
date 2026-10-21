"""Publication identity, lease contracts, and artifact store contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

PUBLICATION_ID_PROPERTY = "flare.publication-id"


def validate_publication_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("publication id must be a non-empty string")
    return value


class PublicationProtocol(StrEnum):
    ICEBERG_SNAPSHOT = "iceberg-snapshot"
    NON_TRANSACTIONAL = "non-transactional"


@dataclass(frozen=True, slots=True)
class PublicationIdentity:
    publication_id: str
    protocol: PublicationProtocol = PublicationProtocol.NON_TRANSACTIONAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            validate_publication_id(self.publication_id)
        except ValueError as exc:
            raise ValueError(
                "publication identity must contain a non-empty id"
            ) from exc
        if not isinstance(self.protocol, PublicationProtocol):
            raise TypeError("protocol must be a PublicationProtocol instance")

    @property
    def value(self) -> str:
        return self.publication_id


@runtime_checkable
class PublicationLeaseStore(Protocol):
    def mark_in_flight(
        self,
        publication_id: str,
        files: list[str],
        protocol: PublicationProtocol,
    ) -> None: ...

    def complete_in_flight(self, publication_id: str) -> None: ...

    def discard_in_flight(self, publication_id: str) -> None: ...


@runtime_checkable
class PublicationPendingStore(PublicationLeaseStore, Protocol):
    def is_pending(self, file_path: str) -> bool: ...


@runtime_checkable
class PublicationResult(Protocol):
    @property
    def total_rows(self) -> int: ...


@runtime_checkable
class PublicationProgressReporter(Protocol):
    def update(
        self,
        current: int,
        total: int,
        rows_processed: int,
        elapsed_batch: float,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PublicationLease:
    files: tuple[str, ...]
    protocol: PublicationProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, PublicationProtocol):
            raise TypeError("publication lease protocol must be PublicationProtocol")
        if not self.files:
            raise ValueError("publication lease files must not be empty")
        if len(set(self.files)) != len(self.files):
            raise ValueError("publication lease files must be unique")

    def to_state(self) -> dict[str, Any]:
        return {"files": list(self.files), "protocol": self.protocol.value}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> PublicationLease:
        if not isinstance(state, Mapping):
            raise TypeError("publication lease state must be an object")
        if set(state) != {"files", "protocol"}:
            raise ValueError(
                "publication lease fields do not match protocol: "
                f"actual={sorted(state)}"
            )
        files = state["files"]
        if not isinstance(files, list) or not all(
            isinstance(path, str) and path for path in files
        ):
            raise ValueError("publication lease files must be non-empty strings")
        try:
            protocol = PublicationProtocol(state["protocol"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported publication protocol: {state['protocol']!r}"
            ) from exc
        return cls(files=tuple(files), protocol=protocol)
