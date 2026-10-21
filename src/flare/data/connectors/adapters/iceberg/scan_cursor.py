"""Typed Iceberg metadata identities and resumable traversal state."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from flare.data.connectors.adapters.iceberg.identity import (
    canonical_digest,
    validate_sha256_digest,
)


def _validate_non_negative(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class ManifestListRef:
    snapshot_id: int
    path: str
    object_length: int
    digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.snapshot_id, int)
            or isinstance(self.snapshot_id, bool)
            or self.snapshot_id <= 0
        ):
            raise ValueError("manifest-list snapshot_id must be positive")
        if (
            not isinstance(self.path, str)
            or not self.path
            or self.path.strip() != self.path
        ):
            raise ValueError("manifest-list path is invalid")
        if (
            not isinstance(self.object_length, int)
            or isinstance(self.object_length, bool)
            or self.object_length <= 0
        ):
            raise ValueError("manifest-list object_length must be positive")
        validate_sha256_digest(self.digest, "manifest-list digest")
        if self.digest != self.compute_digest():
            raise ValueError("manifest-list digest does not match identity")

    def compute_digest(self) -> str:
        return canonical_digest(
            {
                "path": self.path,
                "snapshot_id": self.snapshot_id,
                "object_length": self.object_length,
            }
        )

    @classmethod
    def create(
        cls, *, snapshot_id: int, path: str, object_length: int
    ) -> ManifestListRef:
        value = {
            "path": path,
            "snapshot_id": snapshot_id,
            "object_length": object_length,
        }
        return cls(
            snapshot_id=snapshot_id,
            path=path,
            object_length=object_length,
            digest=canonical_digest(value),
        )

    def encode(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "path": self.path,
            "object_length": self.object_length,
            "digest": self.digest,
        }

    @classmethod
    def decode(cls, value: object) -> ManifestListRef:
        if not isinstance(value, dict) or set(value) != {
            "snapshot_id",
            "path",
            "object_length",
            "digest",
        }:
            raise ValueError("manifest-list ref fields do not match")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ManifestRef:
    path: str
    object_length: int
    content: str
    sequence: int
    added_snapshot_id: int
    partition_spec_id: int
    digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or self.path.strip() != self.path
        ):
            raise ValueError("manifest path is invalid")
        if (
            not isinstance(self.object_length, int)
            or isinstance(self.object_length, bool)
            or self.object_length <= 0
        ):
            raise ValueError("manifest object_length must be positive")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("manifest content is invalid")
        _validate_non_negative(self.sequence, "manifest sequence")
        _validate_non_negative(self.added_snapshot_id, "manifest added snapshot id")
        _validate_non_negative(self.partition_spec_id, "manifest partition spec id")
        validate_sha256_digest(self.digest, "manifest digest")
        if self.digest != self.compute_digest():
            raise ValueError("manifest digest does not match identity")

    def compute_digest(self) -> str:
        return canonical_digest(
            {
                "content": self.content,
                "object_length": self.object_length,
                "path": self.path,
                "sequence": self.sequence,
                "added_snapshot_id": self.added_snapshot_id,
                "partition_spec_id": self.partition_spec_id,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        path: str,
        object_length: int,
        content: str,
        sequence: int,
        added_snapshot_id: int,
        partition_spec_id: int,
    ) -> ManifestRef:
        value = {
            "content": content,
            "object_length": object_length,
            "path": path,
            "sequence": sequence,
            "added_snapshot_id": added_snapshot_id,
            "partition_spec_id": partition_spec_id,
        }
        return cls(**value, digest=canonical_digest(value))

    def encode(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "object_length": self.object_length,
            "content": self.content,
            "sequence": self.sequence,
            "added_snapshot_id": self.added_snapshot_id,
            "partition_spec_id": self.partition_spec_id,
            "digest": self.digest,
        }

    @classmethod
    def decode(cls, value: object) -> ManifestRef:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "object_length",
            "content",
            "sequence",
            "added_snapshot_id",
            "partition_spec_id",
            "digest",
        }:
            raise ValueError("manifest ref fields do not match")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ManifestListOffset:
    block_start: int
    block_index: int
    row_offset: int

    def __post_init__(self) -> None:
        _validate_non_negative(self.block_start, "manifest-list block_start")
        _validate_non_negative(self.block_index, "manifest-list block_index")
        _validate_non_negative(self.row_offset, "manifest-list row_offset")

    def encode(self) -> dict[str, int]:
        return {
            "block_start": self.block_start,
            "block_index": self.block_index,
            "row_offset": self.row_offset,
        }

    @classmethod
    def decode(cls, value: object) -> ManifestListOffset:
        if not isinstance(value, dict) or set(value) != {
            "block_start",
            "block_index",
            "row_offset",
        }:
            raise ValueError("manifest-list offset fields do not match")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ManifestEntryOffset:
    block_start: int
    block_index: int
    row_offset: int

    def __post_init__(self) -> None:
        _validate_non_negative(self.block_start, "manifest block_start")
        _validate_non_negative(self.block_index, "manifest block_index")
        _validate_non_negative(self.row_offset, "manifest row_offset")

    def encode(self) -> dict[str, int]:
        return {
            "block_start": self.block_start,
            "block_index": self.block_index,
            "row_offset": self.row_offset,
        }

    @classmethod
    def decode(cls, value: object) -> ManifestEntryOffset:
        if not isinstance(value, dict) or set(value) != {
            "block_start",
            "block_index",
            "row_offset",
        }:
            raise ValueError("manifest-entry offset fields do not match")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TraversalState:
    snapshot_id: int
    contract_digest: str
    manifest_list: ManifestListRef
    manifest_list_offset: ManifestListOffset
    manifest: ManifestRef | None = None
    manifest_entry_offset: ManifestEntryOffset | None = None
    additions_only: bool = False
    file_record_offset: int = 0

    def __post_init__(self) -> None:
        if self.snapshot_id != self.manifest_list.snapshot_id:
            raise ValueError("traversal snapshot differs from manifest-list ref")
        validate_sha256_digest(self.contract_digest, "contract_digest")
        if (self.manifest is None) != (self.manifest_entry_offset is None):
            raise ValueError("manifest and manifest-entry offset must be paired")
        if not isinstance(self.additions_only, bool):
            raise TypeError("additions_only must be boolean")
        _validate_non_negative(self.file_record_offset, "file record offset")
        if self.manifest is None and self.file_record_offset:
            raise ValueError("file record offset requires a manifest entry")

    @property
    def traversal_position(self) -> tuple[int, int, int, int, int, int]:
        offset = self.manifest_list_offset
        if self.manifest is None:
            return (offset.block_index, offset.row_offset, 0, 0, 0, 0)
        assert self.manifest_entry_offset is not None
        return (
            offset.block_index,
            offset.row_offset,
            1,
            self.manifest_entry_offset.block_index,
            self.manifest_entry_offset.row_offset,
            self.file_record_offset,
        )

    def encode(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "contract_digest": self.contract_digest,
            "manifest_list": self.manifest_list.encode(),
            "manifest_list_offset": self.manifest_list_offset.encode(),
            "manifest": None if self.manifest is None else self.manifest.encode(),
            "manifest_entry_offset": None
            if self.manifest_entry_offset is None
            else self.manifest_entry_offset.encode(),
            "additions_only": self.additions_only,
            "file_record_offset": self.file_record_offset,
        }

    @classmethod
    def decode(cls, value: object) -> TraversalState:
        expected = {
            "snapshot_id",
            "contract_digest",
            "manifest_list",
            "manifest_list_offset",
            "manifest",
            "manifest_entry_offset",
            "additions_only",
            "file_record_offset",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("traversal state fields do not match")
        return cls(
            snapshot_id=value["snapshot_id"],
            contract_digest=value["contract_digest"],
            manifest_list=ManifestListRef.decode(value["manifest_list"]),
            manifest_list_offset=ManifestListOffset.decode(
                value["manifest_list_offset"]
            ),
            manifest=None
            if value["manifest"] is None
            else ManifestRef.decode(value["manifest"]),
            manifest_entry_offset=None
            if value["manifest_entry_offset"] is None
            else ManifestEntryOffset.decode(value["manifest_entry_offset"]),
            additions_only=value["additions_only"],
            file_record_offset=value["file_record_offset"],
        )

    def with_manifest(
        self, manifest: ManifestRef, offset: ManifestEntryOffset
    ) -> TraversalState:
        return replace(
            self, manifest=manifest, manifest_entry_offset=offset, file_record_offset=0
        )

    def with_manifest_offset(self, offset: ManifestEntryOffset) -> TraversalState:
        if self.manifest is None:
            raise ValueError("manifest-entry offset requires a manifest")
        return replace(self, manifest_entry_offset=offset, file_record_offset=0)

    def after_manifest(self, *, next_offset: ManifestListOffset) -> TraversalState:
        return replace(
            self,
            manifest_list_offset=next_offset,
            manifest=None,
            manifest_entry_offset=None,
            file_record_offset=0,
        )


@dataclass(frozen=True, slots=True)
class IcebergScanCursor:
    state: TraversalState
    digest: str = ""

    def __post_init__(self) -> None:
        expected = self.compute_digest()
        if not self.digest:
            object.__setattr__(self, "digest", expected)
        elif self.digest != expected:
            raise ValueError("Iceberg cursor digest does not match state")

    @property
    def snapshot_id(self) -> int:
        return self.state.snapshot_id

    @property
    def contract_digest(self) -> str:
        return self.state.contract_digest

    @property
    def traversal_position(self) -> tuple[int, int, int, int, int, int]:
        return self.state.traversal_position

    def is_after(self, prior: IcebergScanCursor) -> bool:
        if not isinstance(prior, IcebergScanCursor):
            raise TypeError("prior cursor must be IcebergScanCursor")
        if self.snapshot_id != prior.snapshot_id:
            raise ValueError("cursor snapshot identity differs")
        return self.traversal_position > prior.traversal_position

    def advance_records(self, records: int) -> IcebergScanCursor:
        _validate_non_negative(records, "consumed records")
        if self.state.manifest is None:
            raise ValueError("record advancement requires a manifest entry")
        return IcebergScanCursor(
            replace(
                self.state,
                file_record_offset=self.state.file_record_offset + records,
            )
        )

    def compute_digest(self) -> str:
        return canonical_digest({"state": self.state.encode()})

    def encode(self) -> bytes:
        return json.dumps(
            {"state": self.state.encode(), "digest": self.digest},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def decode(cls, payload: bytes) -> IcebergScanCursor:
        if not isinstance(payload, bytes):
            raise TypeError("Iceberg cursor payload must be bytes")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Iceberg cursor payload is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != {"state", "digest"}:
            raise ValueError("Iceberg cursor fields do not match")
        return cls(state=TraversalState.decode(value["state"]), digest=value["digest"])

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: int,
        contract_digest: str,
        manifest_list_ref: ManifestListRef,
        additions_only: bool = False,
    ) -> IcebergScanCursor:
        if snapshot_id != manifest_list_ref.snapshot_id:
            raise ValueError("cursor snapshot differs from manifest-list ref")
        state = TraversalState(
            snapshot_id,
            contract_digest,
            manifest_list_ref,
            ManifestListOffset(0, 0, 0),
            additions_only=additions_only,
        )
        return cls(state=state)


__all__ = [
    "IcebergScanCursor",
    "ManifestEntryOffset",
    "ManifestListOffset",
    "ManifestListRef",
    "ManifestRef",
    "TraversalState",
]
