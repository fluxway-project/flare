"""Source-owned bounded Iceberg metadata traversal contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from flare.data.connectors.adapters.iceberg.scan_cursor import IcebergScanCursor
from flare.data.connectors.adapters.iceberg.scan_result import FileScanResult


class BoundedIcebergTraversal(Protocol):
    def scan_next(
        self,
        table: Any,
        *,
        cursor: IcebergScanCursor,
        max_records: int,
        max_units: int,
        max_metadata_block_bytes: int,
        snapshot_id: int,
        projection: tuple[str, ...],
        row_filter: Mapping[str, Any] | None,
    ) -> FileScanResult: ...


__all__ = ["BoundedIcebergTraversal"]
