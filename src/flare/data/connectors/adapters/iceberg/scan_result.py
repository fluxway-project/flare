"""Bounded Iceberg file scan result value protocol."""

from __future__ import annotations

from dataclasses import dataclass

from flare.data.connectors.adapters.iceberg.scan_cursor import IcebergScanCursor


@dataclass(frozen=True, slots=True)
class FileScanRange:
    path: str
    record_count: int
    record_format: str
    cursor: IcebergScanCursor

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("file scan range path must be non-empty")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise ValueError("file scan range record count must be positive")
        if not isinstance(self.record_format, str) or not self.record_format:
            raise ValueError("file scan range format must be non-empty")
        if not isinstance(self.cursor, IcebergScanCursor):
            raise TypeError("file scan range requires a source cursor")
        if self.cursor.state.manifest is None:
            raise ValueError("file scan range cursor must locate a manifest entry")

    @property
    def record_start(self) -> int:
        return self.cursor.state.file_record_offset

    def cursor_after(self, consumed_records: int) -> IcebergScanCursor:
        if (
            type(consumed_records) is not int
            or not 0 <= consumed_records <= self.record_count
        ):
            raise ValueError("consumed records must lie within the file scan range")
        return self.cursor.advance_records(consumed_records)


@dataclass(frozen=True, slots=True)
class FileScanResult:
    files: tuple[FileScanRange, ...]
    next_cursor: IcebergScanCursor
    snapshot_exhausted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple):
            raise TypeError("file scan result files must be a tuple")
        if any(not isinstance(value, FileScanRange) for value in self.files):
            raise TypeError("file scan result requires FileScanRange values")
        if not isinstance(self.next_cursor, IcebergScanCursor):
            raise TypeError("file scan result requires IcebergScanCursor")
        if not isinstance(self.snapshot_exhausted, bool):
            raise TypeError("file scan snapshot_exhausted must be boolean")

    @property
    def record_count(self) -> int:
        return sum(value.record_count for value in self.files)

    def validate_budget(
        self,
        *,
        max_records: int,
        max_units: int | None = None,
    ) -> None:
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or max_records <= 0
        ):
            raise ValueError("max_records must be positive")
        if self.record_count > max_records:
            raise ValueError("file scan result exceeds record budget")
        if max_units is not None:
            if (
                not isinstance(max_units, int)
                or isinstance(max_units, bool)
                or max_units <= 0
            ):
                raise ValueError("max_units must be positive")
            if len(self.files) > max_units:
                raise ValueError("file scan result exceeds unit budget")

    def validate_transition(
        self,
        requested_cursor: IcebergScanCursor,
    ) -> None:
        if not isinstance(requested_cursor, IcebergScanCursor):
            raise TypeError("requested cursor must be IcebergScanCursor")
        if requested_cursor.snapshot_id != self.next_cursor.snapshot_id:
            raise ValueError("file scan result crosses snapshot identity")
        if requested_cursor.contract_digest != self.next_cursor.contract_digest:
            raise ValueError("file scan result crosses source contract identity")
        if not self.snapshot_exhausted and not self.next_cursor.is_after(
            requested_cursor
        ):
            raise ValueError("unfinished snapshot scan did not advance cursor")
