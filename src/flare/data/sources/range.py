from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flare.data.sources.locator import SourceLocator, normalize_locator


def validate_format_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("source record format must be a string")
    if not value or value.strip() != value:
        raise ValueError("source record format must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True)
class SourceRecordRange:
    source_identity: str
    revision_identity: str
    locator: SourceLocator
    record_start: int
    record_count: int
    schema_digest: str
    record_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.locator, SourceLocator):
            raise TypeError("source range locator must be a SourceLocator")
        if not self.source_identity:
            raise ValueError("source_identity must not be empty")
        if not self.revision_identity:
            raise ValueError("revision_identity must not be empty")
        if self.record_start < 0:
            raise ValueError(
                f"record_start must be non-negative, got {self.record_start}"
            )
        if self.record_count <= 0:
            raise ValueError(f"record_count must be positive, got {self.record_count}")
        if not self.schema_digest:
            raise ValueError("schema_digest must not be empty")
        validate_format_id(self.record_format)

    @property
    def record_end(self) -> int:
        return self.record_start + self.record_count

    def slice(self, start: int, count: int) -> SourceRecordRange:
        if start < 0 or count <= 0 or start + count > self.record_count:
            raise ValueError(
                "range slice exceeds source record range: "
                f"start={start} count={count} length={self.record_count}"
            )
        return SourceRecordRange(
            source_identity=self.source_identity,
            revision_identity=self.revision_identity,
            locator=self.locator,
            record_start=self.record_start + start,
            record_count=count,
            schema_digest=self.schema_digest,
            record_format=self.record_format,
        )

    def can_merge(self, other: SourceRecordRange) -> bool:
        if not isinstance(other, SourceRecordRange):
            return False
        return (
            self.source_identity == other.source_identity
            and self.revision_identity == other.revision_identity
            and self.locator == other.locator
            and self.schema_digest == other.schema_digest
            and self.record_format == other.record_format
            and self.record_end == other.record_start
        )

    def merge(self, other: SourceRecordRange) -> SourceRecordRange:
        if not self.can_merge(other):
            raise ValueError(
                "cannot merge non-contiguous or incompatible source record ranges: "
                f"self={self!r} other={other!r}"
            )
        return SourceRecordRange(
            source_identity=self.source_identity,
            revision_identity=self.revision_identity,
            locator=self.locator,
            record_start=self.record_start,
            record_count=self.record_count + other.record_count,
            schema_digest=self.schema_digest,
            record_format=self.record_format,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_identity": self.source_identity,
            "revision_identity": self.revision_identity,
            "locator": normalize_locator(self.locator).to_mapping(),
            "record_start": self.record_start,
            "record_count": self.record_count,
            "schema_digest": self.schema_digest,
            "record_format": self.record_format,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SourceRecordRange:
        expected_fields = {
            "source_identity",
            "revision_identity",
            "locator",
            "record_start",
            "record_count",
            "schema_digest",
            "record_format",
        }
        if set(value) != expected_fields:
            raise ValueError(
                "source record range fields do not match protocol: "
                f"expected={sorted(expected_fields)} actual={sorted(value)}"
            )
        for field in (
            "source_identity",
            "revision_identity",
            "locator",
            "schema_digest",
            "record_format",
        ):
            if field != "locator" and not isinstance(value[field], str):
                raise TypeError(f"source record range {field} must be a string")
        for field in ("record_start", "record_count"):
            if not isinstance(value[field], int) or isinstance(value[field], bool):
                raise TypeError(f"source record range {field} must be an integer")
        return cls(
            source_identity=value["source_identity"],
            revision_identity=value["revision_identity"],
            locator=normalize_locator(value["locator"]),
            record_start=value["record_start"],
            record_count=value["record_count"],
            schema_digest=value["schema_digest"],
            record_format=value["record_format"],
        )


__all__ = ["SourceRecordRange", "validate_format_id"]
