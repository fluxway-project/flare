"""Strict source locator values used by source checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable


@runtime_checkable
class SourceLocator(Protocol):
    kind: str

    def to_mapping(self) -> dict[str, Any]: ...

    def canonical(self) -> str: ...


class FileLocator(str):
    kind = "file"

    def __new__(cls, path: str) -> Self:
        if not isinstance(path, str) or not path or path.strip() != path:
            raise ValueError("file locator path must be a non-empty canonical string")
        if "\x00" in path:
            raise ValueError("file locator path must not contain NUL")
        return str.__new__(cls, path)

    def canonical(self) -> str:
        return str(self)

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": str(self)}


@dataclass(frozen=True, slots=True, eq=False)
class IcebergTaskLocator:
    """Iceberg task identity retaining path compatibility at IO boundaries."""

    kind = "iceberg"
    snapshot_id: int
    data_file: str
    task_digest: str
    scan_spec_digest: str

    def __post_init__(self) -> None:
        if isinstance(self.snapshot_id, bool) or self.snapshot_id < 0:
            raise ValueError("Iceberg locator snapshot_id must be non-negative")
        for name, value in (
            ("data_file", self.data_file),
            ("task_digest", self.task_digest),
            ("scan_spec_digest", self.scan_spec_digest),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"Iceberg locator {name} must be canonical")

    def __str__(self) -> str:
        return self.data_file

    def canonical(self) -> str:
        return self.data_file

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IcebergTaskLocator):
            return (
                self.snapshot_id,
                self.data_file,
                self.task_digest,
                self.scan_spec_digest,
            ) == (
                other.snapshot_id,
                other.data_file,
                other.task_digest,
                other.scan_spec_digest,
            )
        if isinstance(other, str):
            return self.data_file == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.data_file)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return (
            self.__class__,
            (
                self.snapshot_id,
                self.data_file,
                self.task_digest,
                self.scan_spec_digest,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "snapshot_id": self.snapshot_id,
            "data_file": self.data_file,
            "task_digest": self.task_digest,
            "scan_spec_digest": self.scan_spec_digest,
        }


def normalize_locator(value: SourceLocator | str | Mapping[str, Any]) -> SourceLocator:
    if isinstance(value, (FileLocator, IcebergTaskLocator)):
        return value
    if isinstance(value, str):
        return FileLocator(value)
    if not isinstance(value, Mapping):
        raise TypeError("source locator must be a typed locator or mapping")
    kind = value.get("kind")
    if kind == "file":
        if set(value) != {"kind", "path"}:
            raise ValueError("file locator fields do not match protocol")
        return FileLocator(value["path"])
    if kind == "iceberg":
        expected = {
            "kind",
            "snapshot_id",
            "data_file",
            "task_digest",
            "scan_spec_digest",
        }
        if set(value) != expected:
            raise ValueError("Iceberg locator fields do not match protocol")
        return IcebergTaskLocator(
            snapshot_id=value["snapshot_id"],
            data_file=value["data_file"],
            task_digest=value["task_digest"],
            scan_spec_digest=value["scan_spec_digest"],
        )
    raise ValueError(f"unsupported source locator kind: {kind!r}")
