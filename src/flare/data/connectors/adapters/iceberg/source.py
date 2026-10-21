"""Iceberg source adapter for snapshot-delta discovery."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import pyarrow as pa
from pyiceberg.table.snapshots import Operation

from flare.data.connectors.adapters.iceberg.catalog import (
    load_iceberg_table,
    normalize_iceberg_catalog_kwargs,
)
from flare.data.connectors.adapters.iceberg.identity import (
    bound_source_identity,
    canonical_digest,
    require_table_uuid,
    sha256_bytes,
    validate_table_uuid,
)
from flare.data.connectors.adapters.iceberg.manifest_stream import (
    IcebergManifestTraversal,
)
from flare.data.connectors.adapters.iceberg.modes import (
    ContinuousAppend,
    PinnedSnapshot,
)
from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestListRef,
)
from flare.data.connectors.adapters.iceberg.scan_result import (
    FileScanResult,
)
from flare.data.connectors.adapters.iceberg.traversal import BoundedIcebergTraversal
from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange
from flare.data.schema.feature_catalog import validate_storage_columns
from flare.data.sources.locator import IcebergTaskLocator

_BINDING_IDENTITY_FIELDS = frozenset(
    {
        "configuration_identity",
        "source_identity",
        "mode_identity",
        "native_identity",
    }
)


def _binding_identity_mapping(
    *,
    configuration_identity: str,
    source_identity: str,
    mode_identity: str,
    native_identity: str,
) -> Mapping[str, str]:
    values = {
        "configuration_identity": configuration_identity,
        "source_identity": source_identity,
        "mode_identity": mode_identity,
        "native_identity": native_identity,
    }
    if set(values) != _BINDING_IDENTITY_FIELDS or any(
        not isinstance(value, str) or not value or value.strip() != value
        for value in values.values()
    ):
        raise ValueError("Iceberg binding identity values must be canonical strings")
    return MappingProxyType(values)


def _normalize_record_format(raw_format: Any) -> str:
    if raw_format is None:
        raise ValueError("Iceberg data file has no declared file format")
    value = getattr(raw_format, "value", raw_format)
    normalized = str(value).lower()
    if not normalized or normalized.strip() != normalized:
        raise ValueError(f"Iceberg data file has invalid format: {raw_format!r}")
    return normalized


def _mode_identity(
    mode: PinnedSnapshot | ContinuousAppend | None,
) -> Mapping[str, Any] | None:
    if mode is None:
        return None
    if isinstance(mode, PinnedSnapshot):
        return {"kind": "pinned", "snapshot_id": mode.snapshot_id}
    if isinstance(mode, ContinuousAppend):
        return {
            "kind": "continuous",
            "branch": mode.branch,
        }
    raise TypeError("unsupported Iceberg source mode")


def _catalog_identity_kwargs(catalog_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return catalog settings that define data identity, excluding credentials."""

    return {
        key: value
        for key, value in catalog_kwargs.items()
        if key not in {"token", "header.Authorization", "auth"}
    }


@dataclass(frozen=True, slots=True)
class _IcebergSourceConfig:
    table_identifier: str
    catalog_kwargs: dict[str, Any]
    projection: tuple[str, ...]
    source_identity: str
    schema_digest: str

    @classmethod
    def create(
        cls,
        *,
        table_identifier: str,
        catalog_kwargs: Mapping[str, Any],
        projection: Sequence[str],
        mode_identity: Mapping[str, Any] | None = None,
        row_filter: Mapping[str, Any] | None = None,
    ) -> _IcebergSourceConfig:
        if not table_identifier:
            raise ValueError("table_identifier must not be empty")
        if not projection:
            raise ValueError("projection must not be empty")
        normalized_projection = validate_storage_columns(projection)
        normalized_catalog = normalize_iceberg_catalog_kwargs(catalog_kwargs)
        identity_digest = canonical_digest(
            {
                "catalog": _catalog_identity_kwargs(normalized_catalog),
                "projection": normalized_projection,
                "table_identifier": table_identifier,
                "mode": None if mode_identity is None else dict(mode_identity),
                "row_filter": None if row_filter is None else dict(row_filter),
            }
        )
        return cls(
            table_identifier=table_identifier,
            catalog_kwargs=normalized_catalog,
            projection=normalized_projection,
            source_identity=(
                f"iceberg://{table_identifier}?identity={identity_digest}"
            ),
            schema_digest=canonical_digest(
                normalized_projection,
                sort_keys=False,
            ),
        )


class IcebergSourcePlanner:
    """Plans file ranges through bounded native snapshot traversal."""

    def __init__(
        self,
        *,
        table_identifier: str,
        catalog_kwargs: Mapping[str, Any],
        projection: Sequence[str],
        row_filter: Any | None = None,
        mode: PinnedSnapshot | ContinuousAppend | None = None,
        traversal: BoundedIcebergTraversal | None = None,
        metadata_scan_records: int = 1024,
        metadata_block_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if row_filter is not None and not isinstance(row_filter, Mapping):
            raise TypeError("Iceberg row_filter must be a canonical mapping or None")
        if mode is not None and not isinstance(
            mode, (PinnedSnapshot, ContinuousAppend)
        ):
            raise TypeError(
                "Iceberg source mode must be PinnedSnapshot or ContinuousAppend"
            )
        mode_identity = _mode_identity(mode)
        config = _IcebergSourceConfig.create(
            table_identifier=table_identifier,
            catalog_kwargs=catalog_kwargs,
            projection=projection,
            mode_identity=mode_identity,
            row_filter=row_filter,
        )
        self.source_identity = config.source_identity
        self._base_source_identity = config.source_identity
        self.schema_digest = config.schema_digest
        self._table_identifier = config.table_identifier
        self._catalog_kwargs = config.catalog_kwargs
        self._projection = config.projection
        self._row_filter = row_filter
        self._mode = mode
        self._traversal = IcebergManifestTraversal() if traversal is None else traversal
        if (
            not isinstance(metadata_scan_records, int)
            or isinstance(metadata_scan_records, bool)
            or metadata_scan_records <= 0
        ):
            raise ValueError("metadata_scan_records must be positive")
        if (
            not isinstance(metadata_block_bytes, int)
            or isinstance(metadata_block_bytes, bool)
            or metadata_block_bytes <= 0
        ):
            raise ValueError("metadata_block_bytes must be positive")
        self._metadata_scan_records = metadata_scan_records
        self._metadata_block_bytes = metadata_block_bytes
        self._table: Any | None = None
        self._table_uuid: str | None = None

    @property
    def configuration_identity(self) -> str:
        return self._base_source_identity

    @property
    def mode_identity(self) -> str:
        return canonical_digest(_mode_identity(self._mode) or {"kind": "current"})

    @property
    def binding_identity(self) -> Mapping[str, str]:
        self._current_table()
        return _binding_identity_mapping(
            configuration_identity=self.configuration_identity,
            source_identity=self.source_identity,
            mode_identity=self.mode_identity,
            native_identity=self.table_uuid,
        )

    @property
    def table_uuid(self) -> str:
        self._current_table()
        if self._table_uuid is None:
            raise RuntimeError("Iceberg table UUID was not bound")
        return self._table_uuid

    def record_range_for_file(
        self,
        *,
        snapshot_id: int,
        path: str,
        record_start: int,
        record_count: int,
        record_format: str,
    ) -> SourceRecordRange:
        self._current_table()
        if (
            not isinstance(snapshot_id, int)
            or isinstance(snapshot_id, bool)
            or snapshot_id <= 0
        ):
            raise ValueError("snapshot_id must be positive")
        if not isinstance(path, str) or not path:
            raise ValueError("data file path must be non-empty")
        if (
            not isinstance(record_count, int)
            or isinstance(record_count, bool)
            or record_count <= 0
        ):
            raise ValueError("data file record count must be positive")
        normalized_format = _normalize_record_format(record_format)
        return SourceRecordRange(
            source_identity=self.source_identity,
            revision_identity=str(snapshot_id),
            locator=self._locator(path, snapshot_id),
            record_start=record_start,
            record_count=record_count,
            schema_digest=self.schema_digest,
            record_format=normalized_format,
        )

    @property
    def is_finite(self) -> bool:
        return not isinstance(self._mode, ContinuousAppend)

    def scan_next(
        self,
        cursor: IcebergScanCursor,
        *,
        max_records: int | None = None,
        max_units: int | None = None,
    ) -> FileScanResult:
        """Capture exactly one bounded Iceberg file scan result."""

        if not isinstance(cursor, IcebergScanCursor):
            raise TypeError("Iceberg scan capture requires IcebergScanCursor")
        self._current_table()
        expected_contract = sha256_bytes(self.source_identity.encode("utf-8"))
        if cursor.contract_digest != expected_contract:
            raise ValueError("Iceberg scan cursor contract does not match source")
        records_limit = (
            self._metadata_scan_records if max_records is None else max_records
        )
        units_limit = records_limit if max_units is None else max_units
        for value, name in (
            (records_limit, "max_records"),
            (units_limit, "max_units"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        table = self._current_table()
        snapshot = table.snapshot_by_id(cursor.snapshot_id)
        if snapshot is None:
            raise ValueError(f"Iceberg snapshot does not exist: {cursor.snapshot_id}")
        if cursor.state.additions_only:
            self._validate_append_snapshot(snapshot)
        scan_result = self._traversal.scan_next(
            table,
            cursor=cursor,
            max_records=records_limit,
            max_units=units_limit,
            max_metadata_block_bytes=self._metadata_block_bytes,
            snapshot_id=cursor.snapshot_id,
            projection=self._projection,
            row_filter=self._row_filter,
        )
        scan_result.validate_budget(
            max_records=records_limit,
            max_units=units_limit,
        )
        scan_result.validate_transition(cursor)
        if not self.is_finite:
            return self._scan_continuous(
                cursor=cursor,
                scan_result=scan_result,
            )
        return scan_result

    def _scan_continuous(
        self,
        *,
        cursor: IcebergScanCursor,
        scan_result: FileScanResult,
    ) -> FileScanResult:
        if scan_result.files or not scan_result.snapshot_exhausted:
            return scan_result
        table = self._current_table(refresh=True)
        snapshot = table.snapshot_by_id(cursor.snapshot_id)
        if snapshot is None:
            raise ValueError(f"Iceberg snapshot disappeared: {cursor.snapshot_id}")
        latest = self._branch_snapshot(table)
        if latest is None:
            raise ValueError("Iceberg source branch disappeared")
        following = self._next_snapshot(
            table, current_snapshot=snapshot, latest_snapshot=latest
        )
        if following is None:
            return scan_result
        self._validate_append_snapshot(following)
        return replace(
            scan_result,
            next_cursor=self._cursor_for_snapshot(
                table, following, additions_only=True
            ),
            snapshot_exhausted=False,
        )

    def _branch_snapshot(self, table: Any) -> Any:
        if not isinstance(self._mode, ContinuousAppend):
            raise TypeError("branch selection requires ContinuousAppend")
        return table.snapshot_by_name(self._mode.branch)

    @staticmethod
    def _validate_append_snapshot(snapshot: Any) -> None:
        if (
            snapshot.summary is None
            or snapshot.summary.operation is not Operation.APPEND
        ):
            raise ValueError("continuous Iceberg consumption requires append snapshots")

    def _cursor_for_snapshot(
        self, table: Any, snapshot: Any, *, additions_only: bool
    ) -> IcebergScanCursor:
        return IcebergScanCursor.create(
            snapshot_id=snapshot.snapshot_id,
            contract_digest=sha256_bytes(self.source_identity.encode("utf-8")),
            manifest_list_ref=ManifestListRef.create(
                snapshot_id=snapshot.snapshot_id,
                path=snapshot.manifest_list,
                object_length=len(table.io.new_input(snapshot.manifest_list)),
            ),
            additions_only=additions_only,
        )

    @staticmethod
    def _next_snapshot(
        table: Any, *, current_snapshot: Any, latest_snapshot: Any
    ) -> Any | None:
        current_id = int(current_snapshot.snapshot_id)
        latest_id = int(latest_snapshot.snapshot_id)
        if current_id == latest_id:
            return None
        child = latest_snapshot
        seen: set[int] = set()
        while True:
            child_id = int(child.snapshot_id)
            if child_id in seen:
                raise ValueError("Iceberg snapshot parent chain contains a cycle")
            seen.add(child_id)
            parent_id = getattr(child, "parent_snapshot_id", None)
            if parent_id is None:
                raise ValueError("Iceberg latest snapshot has no parent identity")
            parent_id = int(parent_id)
            if parent_id == current_id:
                return child
            child = table.snapshot_by_id(parent_id)
            if child is None:
                raise ValueError(f"Iceberg snapshot parent is unavailable: {parent_id}")

    def initial_scan_cursor(self) -> IcebergScanCursor | None:
        table = self._current_table(refresh=True)
        if isinstance(self._mode, PinnedSnapshot):
            snapshot = table.snapshot_by_id(self._mode.snapshot_id)
        elif isinstance(self._mode, ContinuousAppend):
            snapshot = self._branch_snapshot(table)
        else:
            snapshot = table.current_snapshot()
        if snapshot is None:
            if isinstance(self._mode, ContinuousAppend):
                return None
            raise ValueError("Iceberg source has no snapshot for scan capture")
        return self._cursor_for_snapshot(table, snapshot, additions_only=False)

    def _current_table(self, *, refresh: bool = False) -> Any:
        if self._table is None:
            self._table = load_iceberg_table(
                self._catalog_kwargs,
                self._table_identifier,
            )
            self._bind_table_identity(self._table)
        elif self._table_uuid is None:
            self._bind_table_identity(self._table)
        if refresh:
            refreshed = self._table.refresh()
            self._validate_table_uuid(refreshed)
            self._table = refreshed
        if self._table is None:
            raise TypeError("Iceberg table load or refresh returned None")
        return self._table

    def _bind_table_identity(self, table: Any) -> None:
        normalized = require_table_uuid(table)
        self._table_uuid = normalized
        self.source_identity = bound_source_identity(
            self._base_source_identity, normalized
        )

    def _validate_table_uuid(self, table: Any) -> None:
        validate_table_uuid(table, self._table_uuid)

    def _locator(self, path: str, snapshot_id: int) -> IcebergTaskLocator:
        payload = json.dumps(
            {
                "data_file": path,
                "projection": self._projection,
                "row_filter": None
                if self._row_filter is None
                else dict(self._row_filter),
                "snapshot_id": snapshot_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = sha256_bytes(payload)
        return IcebergTaskLocator(
            snapshot_id=snapshot_id,
            data_file=path,
            task_digest=digest,
            scan_spec_digest=self.schema_digest,
        )


class IcebergSourceReader:
    def __init__(
        self,
        *,
        table_identifier: str,
        catalog_kwargs: Mapping[str, Any],
        projection: Sequence[str],
        format_readers: Mapping[str, Any],
        row_filter: Mapping[str, Any] | None = None,
        mode: PinnedSnapshot | ContinuousAppend | None = None,
    ) -> None:
        if row_filter is not None and not isinstance(row_filter, Mapping):
            raise TypeError("Iceberg row_filter must be a canonical mapping or None")
        if mode is not None and not isinstance(
            mode, (PinnedSnapshot, ContinuousAppend)
        ):
            raise TypeError(
                "Iceberg source mode must be PinnedSnapshot or ContinuousAppend"
            )
        config = _IcebergSourceConfig.create(
            table_identifier=table_identifier,
            catalog_kwargs=catalog_kwargs,
            projection=projection,
            mode_identity=_mode_identity(mode),
            row_filter=row_filter,
        )
        if not format_readers:
            raise ValueError("Iceberg source format_readers must not be empty")
        normalized_readers: dict[str, Any] = {}
        for format_id, reader in format_readers.items():
            normalized = _normalize_record_format(format_id)
            if normalized in normalized_readers:
                raise ValueError(f"duplicate Iceberg format reader: {normalized}")
            if not callable(getattr(reader, "create_session", None)):
                raise TypeError(
                    f"Iceberg format reader must create a closeable session: {normalized}"
                )
            normalized_readers[normalized] = reader
        self.source_identity = config.source_identity
        self._base_source_identity = config.source_identity
        self.schema_digest = config.schema_digest
        self._table_identifier = config.table_identifier
        self._catalog_kwargs = config.catalog_kwargs
        self._projection = config.projection
        self._mode = mode
        self._format_readers = MappingProxyType(normalized_readers)
        self._format_sessions: dict[str, Any] = {}
        self._closed = False
        self._session_lock = threading.Lock()
        self._table: Any | None = None
        self._table_uuid: str | None = None
        self._expected_source_identity: str | None = None
        self._materialization_revision: str | None = None
        self._table_lock = threading.Lock()

    @property
    def configuration_identity(self) -> str:
        return self._base_source_identity

    @property
    def mode_identity(self) -> str:
        return canonical_digest(_mode_identity(self._mode) or {"kind": "current"})

    @property
    def table_uuid(self) -> str:
        self._ensure_table_bound()
        if self._table_uuid is None:
            raise RuntimeError("Iceberg table UUID was not bound")
        return self._table_uuid

    @property
    def binding_identity(self) -> Mapping[str, str]:
        self._ensure_table_bound()
        return _binding_identity_mapping(
            configuration_identity=self.configuration_identity,
            source_identity=self.source_identity,
            mode_identity=self.mode_identity,
            native_identity=self.table_uuid,
        )

    @property
    def schema(self) -> pa.Schema:
        self._ensure_table_bound()
        if self._table is None:
            raise RuntimeError("Iceberg table was not bound")
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        full_schema = schema_to_pyarrow(self._table.schema())
        if self._projection:
            fields = [full_schema.field(name) for name in self._projection]
            return pa.schema(fields, metadata=full_schema.metadata)
        return full_schema

    def bind_source_identity(self, source_identity: str) -> None:
        if (
            not isinstance(source_identity, str)
            or not source_identity
            or source_identity.strip() != source_identity
        ):
            raise ValueError("Iceberg reader source identity must be canonical")
        if (
            self._expected_source_identity is not None
            and self._expected_source_identity != source_identity
        ):
            raise ValueError("Iceberg reader source identity changed during session")
        if self._table_uuid is not None and self.source_identity != source_identity:
            raise ValueError("Iceberg reader source identity differs from assignment")
        self._expected_source_identity = source_identity

    def close(self) -> None:
        with self._session_lock:
            if self._closed:
                return
            self._closed = True
            sessions = tuple(self._format_sessions.values())
            self._format_sessions.clear()
        for session in sessions:
            session.close()

    def _format_session(self, record_format: str) -> Any:
        with self._session_lock:
            if self._closed:
                raise RuntimeError("Iceberg source reader is closed")
            session = self._format_sessions.get(record_format)
            if session is None:
                factory = self._format_readers[record_format]
                session = factory.create_session(source_identity=self.source_identity)
                self._format_sessions[record_format] = session
            return session

    def read(self, record_range: SourceRecordRange, projection: Sequence[str]):
        if self._closed:
            raise RuntimeError("Iceberg source reader is closed")
        self._ensure_table_bound()
        if record_range.source_identity != self.source_identity:
            raise ValueError("record range belongs to another source")
        if record_range.schema_digest != self.schema_digest:
            raise ValueError("record range schema differs from Iceberg source reader")
        if tuple(projection) != self._projection:
            raise ValueError(
                "source range read projection differs from configured projection: "
                f"configured={self._projection!r} requested={tuple(projection)!r}"
            )
        if record_range.record_format not in self._format_readers:
            raise ValueError(
                "Iceberg record format has no registered reader: "
                f"{record_range.record_format!r}"
            )
        batches = self.read_batches(
            record_range, projection, batch_size=record_range.record_count
        )
        return pa.Table.from_batches(list(batches))

    def read_batches(
        self,
        record_range: SourceRecordRange,
        projection: Sequence[str],
        *,
        batch_size: int,
    ):
        if self._closed:
            raise RuntimeError("Iceberg source reader is closed")
        self._ensure_table_bound()
        if record_range.source_identity != self.source_identity:
            raise ValueError("record range belongs to another source")
        if tuple(projection) != self._projection:
            raise ValueError(
                "source range read projection differs from configured projection"
            )
        if record_range.record_format not in self._format_readers:
            raise ValueError(
                "Iceberg record format has no registered reader: "
                f"{record_range.record_format!r}"
            )
        batches = self._format_session(record_range.record_format).read_batches(
            revision_identity=record_range.revision_identity,
            locator=record_range.locator.canonical(),
            input_file_factory=lambda: self._input_file(record_range),
            projection=self._projection,
            row_start=record_range.record_start,
            row_count=record_range.record_count,
            batch_size=batch_size,
        )
        if not hasattr(batches, "__iter__") and not hasattr(batches, "__aiter__"):
            raise TypeError(
                "Iceberg read_batches must return an iterable RecordBatch stream"
            )
        return batches

    def _input_file(self, record_range: SourceRecordRange) -> Any:
        try:
            snapshot_id = int(record_range.revision_identity)
        except ValueError as exc:
            raise ValueError(
                "Iceberg record range revision must be a snapshot id"
            ) from exc
        with self._table_lock:
            if self._table is None:
                self._table = load_iceberg_table(
                    self._catalog_kwargs,
                    self._table_identifier,
                )
                self._bind_table_identity(self._table)
            elif self._materialization_revision != record_range.revision_identity:
                refreshed = self._table.refresh()
                self._validate_table_uuid(refreshed)
                self._table = refreshed
            if self._table is None:
                raise TypeError("Iceberg table load or refresh returned None")
            if self._table.snapshot_by_id(snapshot_id) is None:
                raise ValueError(f"Iceberg snapshot does not exist: {snapshot_id}")
            self._materialization_revision = record_range.revision_identity
            return self._table.io.new_input(record_range.locator.canonical())

    def _ensure_table_bound(self) -> None:
        with self._table_lock:
            if self._table is None:
                self._table = load_iceberg_table(
                    self._catalog_kwargs,
                    self._table_identifier,
                )
                self._bind_table_identity(self._table)
            else:
                self._validate_table_uuid(self._table)

    def _bind_table_identity(self, table: Any) -> None:
        normalized = require_table_uuid(table)
        self._table_uuid = normalized
        self.source_identity = bound_source_identity(
            self._base_source_identity, normalized
        )
        if (
            self._expected_source_identity is not None
            and self.source_identity != self._expected_source_identity
        ):
            raise ValueError("Iceberg reader source identity differs from assignment")

    def _validate_table_uuid(self, table: Any) -> None:
        validate_table_uuid(table, self._table_uuid)
