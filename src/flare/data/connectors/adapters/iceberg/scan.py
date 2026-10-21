from __future__ import annotations

from collections.abc import Mapping

from flare.data.connectors.adapters.iceberg.scan_cursor import IcebergScanCursor
from flare.data.connectors.adapters.iceberg.scan_result import FileScanResult
from flare.data.connectors.adapters.iceberg.source import IcebergSourcePlanner
from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
)
from fluxway.common.metadata import metadata_digest, pack_metadata
from fluxway.control_plane.ingestion import LayoutUnitDescriptor


def iceberg_scan_lineage_identity(cursor: IcebergScanCursor) -> str:
    if not isinstance(cursor, IcebergScanCursor):
        raise TypeError("Iceberg lineage identity requires an IcebergScanCursor")
    return metadata_digest(
        {
            "additions_only": cursor.state.additions_only,
            "manifest_list_digest": cursor.state.manifest_list.digest,
            "snapshot_id": cursor.snapshot_id,
        }
    )


class IcebergSourceScanAdapter:
    def __init__(self, planner: IcebergSourcePlanner) -> None:
        if not isinstance(planner, IcebergSourcePlanner):
            raise TypeError("Iceberg scan adapter requires IcebergSourcePlanner")
        self._planner = planner

    @property
    def binding_identity(self) -> Mapping[str, str]:
        return self._planner.binding_identity

    def initial_scan_cursor(self) -> OpaqueSourceCursor | None:
        cursor = self._planner.initial_scan_cursor()
        if cursor is None:
            return None
        return OpaqueSourceCursor.from_payload(cursor.encode())

    @staticmethod
    def advance_cursor(
        cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor:
        native = IcebergScanCursor.decode(cursor.payload)
        return OpaqueSourceCursor.from_payload(native.advance_records(records).encode())

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult:
        if not isinstance(cursor, OpaqueSourceCursor):
            raise TypeError("Iceberg scan adapter requires an opaque cursor")
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("Iceberg scan adapter requires a typed scan budget")
        iceberg_cursor = IcebergScanCursor.decode(cursor.payload)
        scan_result = self._planner.scan_next(
            iceberg_cursor,
            max_records=budget.max_records,
            max_units=budget.max_units,
        )
        if not isinstance(scan_result, FileScanResult):
            raise TypeError("Iceberg traversal must return FileScanResult")
        units: list[LayoutUnitDescriptor] = []
        for file_range in scan_result.files:
            record_count = file_range.record_count
            record_range = self._planner.record_range_for_file(
                snapshot_id=file_range.cursor.snapshot_id,
                path=file_range.path,
                record_start=file_range.record_start,
                record_count=record_count,
                record_format=file_range.record_format,
            )
            units.append(
                LayoutUnitDescriptor(
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
            )
        next_cursor = OpaqueSourceCursor.from_payload(scan_result.next_cursor.encode())
        return SourceScanResult(
            source_identity=self._planner.source_identity,
            view_identity=str(iceberg_cursor.snapshot_id),
            schema_digest=self._planner.schema_digest,
            lineage_digest=iceberg_scan_lineage_identity(iceberg_cursor),
            units=tuple(units),
            unit_cursors=tuple(
                OpaqueSourceCursor.from_payload(file_range.cursor.encode())
                for file_range in scan_result.files
            ),
            records=sum(unit.cardinality for unit in units),
            next_cursor=next_cursor,
            status=SourceScanStatus.from_scan(
                records=scan_result.record_count,
                exhausted=scan_result.snapshot_exhausted,
                finite=self._planner.is_finite,
            ),
        )


__all__ = ["IcebergSourceScanAdapter", "iceberg_scan_lineage_identity"]
