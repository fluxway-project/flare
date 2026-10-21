from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from flare.data.connectors.adapters.file.source import FileListSourceAdapter
from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange
from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
)
from fluxway.common.metadata import metadata_digest, pack_metadata
from fluxway.control_plane.ingestion import LayoutUnitDescriptor


class FileListSourceScanAdapter:
    def __init__(self, source: FileListSourceAdapter) -> None:
        if not isinstance(source, FileListSourceAdapter):
            raise TypeError("file scan adapter requires a FileListSourceAdapter")
        self._source = source
        self._ranges = source.record_ranges()
        if not self._ranges:
            raise ValueError("file scan adapter requires at least one source range")
        self._lineage_digest = metadata_digest(
            {
                "revision_identity": source.revision_identity,
                "source_identity": source.source_identity,
            }
        )

    @property
    def binding_identity(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "configuration_identity": self._source.source_identity,
                "source_identity": self._source.source_identity,
                "mode_identity": "static",
                "native_identity": self._source.revision_identity,
            }
        )

    @property
    def schema_digest(self) -> str:
        return self._source.schema_digest

    @property
    def is_finite(self) -> bool:
        return True

    def initial_scan_cursor(self) -> OpaqueSourceCursor:
        return self._cursor(0, 0)

    def advance_cursor(
        self, cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor:
        index, offset = self._decode_cursor(cursor)
        if not isinstance(records, int) or isinstance(records, bool) or records < 0:
            raise ValueError("file scan cursor advancement must be non-negative")
        remaining = records
        while remaining:
            if index == len(self._ranges):
                raise ValueError("file scan cursor advancement exceeds source")
            available = self._ranges[index].record_count - offset
            consumed = min(available, remaining)
            offset += consumed
            remaining -= consumed
            if offset == self._ranges[index].record_count:
                index += 1
                offset = 0
        return self._cursor(index, offset)

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult:
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("file scan adapter requires a typed scan budget")
        index, offset = self._decode_cursor(cursor)
        units: list[LayoutUnitDescriptor] = []
        unit_cursors: list[OpaqueSourceCursor] = []
        records = 0
        while (
            index < len(self._ranges)
            and len(units) < budget.max_units
            and records < budget.max_records
        ):
            source_range = self._ranges[index]
            available = source_range.record_count - offset
            count = min(available, budget.max_records - records)
            if count <= 0:
                break
            unit_cursor = self._cursor(index, offset)
            record_range = source_range.slice(offset, count)
            units.append(self._unit_descriptor(record_range))
            unit_cursors.append(unit_cursor)
            records += count
            offset += count
            if offset == source_range.record_count:
                index += 1
                offset = 0

        next_cursor = self._cursor(index, offset)
        exhausted = index == len(self._ranges)
        return SourceScanResult(
            source_identity=self._source.source_identity,
            view_identity=self._source.revision_identity,
            schema_digest=self._source.schema_digest,
            lineage_digest=self._lineage_digest,
            units=tuple(units),
            unit_cursors=tuple(unit_cursors),
            records=records,
            next_cursor=next_cursor,
            status=SourceScanStatus.from_scan(
                records=records,
                exhausted=exhausted,
                finite=self.is_finite,
            ),
        )

    def _unit_descriptor(self, record_range: SourceRecordRange) -> LayoutUnitDescriptor:
        return LayoutUnitDescriptor(
            unit_id=metadata_digest(record_range.to_dict()),
            revision_identity=record_range.revision_identity,
            cardinality=record_range.record_count,
            cardinality_metric="rows",
            split_policy="SplittableSpan",
            locality_key=record_range.locator.canonical(),
            physical_identity=record_range.locator.canonical(),
            logical_start=record_range.record_start,
            opaque_token=pack_metadata(record_range.to_dict()),
        )

    def _cursor(self, index: int, offset: int) -> OpaqueSourceCursor:
        payload = json.dumps(
            {
                "format": "file-list-v1",
                "file_index": index,
                "record_offset": offset,
                "source_identity": self._source.source_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return OpaqueSourceCursor.from_payload(payload)

    def _decode_cursor(self, cursor: OpaqueSourceCursor) -> tuple[int, int]:
        if not isinstance(cursor, OpaqueSourceCursor):
            raise TypeError("file scan adapter requires an opaque cursor")
        try:
            value: Any = json.loads(cursor.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("file scan cursor payload is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != {
            "format",
            "file_index",
            "record_offset",
            "source_identity",
        }:
            raise ValueError("file scan cursor fields do not match")
        if value["format"] != "file-list-v1":
            raise ValueError("file scan cursor format does not match")
        if value["source_identity"] != self._source.source_identity:
            raise ValueError("file scan cursor source identity does not match")
        index = value["file_index"]
        offset = value["record_offset"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index > len(self._ranges)
        ):
            raise ValueError("file scan cursor file index is invalid")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("file scan cursor record offset is invalid")
        if index == len(self._ranges):
            if offset != 0:
                raise ValueError("terminal file scan cursor offset must be zero")
        elif offset >= self._ranges[index].record_count:
            raise ValueError("file scan cursor record offset exceeds source range")
        return index, offset


__all__ = ["FileListSourceScanAdapter"]
