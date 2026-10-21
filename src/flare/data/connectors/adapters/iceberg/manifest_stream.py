"""Bounded, resumable Iceberg manifest metadata traversal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pyiceberg.avro.resolver import resolve_reader
from pyiceberg.manifest import (
    MANIFEST_ENTRY_SCHEMAS,
    MANIFEST_LIST_FILE_SCHEMAS,
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
    _inherit_from_manifest,
)

from flare.data.connectors.adapters.iceberg.avro_blocks import BoundedAvroBlocks
from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestEntryOffset,
    ManifestListOffset,
    ManifestListRef,
    ManifestRef,
    TraversalState,
)
from flare.data.connectors.adapters.iceberg.scan_result import (
    FileScanRange,
    FileScanResult,
)


class IcebergManifestTraversal:
    """Advance through at most one native metadata block per call."""

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
    ) -> FileScanResult:
        del projection
        if row_filter is not None:
            raise ValueError("Iceberg manifest traversal does not support row filters")
        if snapshot_id != cursor.snapshot_id:
            raise ValueError("manifest traversal snapshot identity mismatch")
        for value in (max_records, max_units, max_metadata_block_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("manifest traversal limits must be positive integers")
        snapshot = table.snapshot_by_id(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Iceberg snapshot does not exist: {snapshot_id}")
        io = table.io
        format_version = self._format_version(table)
        if (
            format_version not in MANIFEST_LIST_FILE_SCHEMAS
            or format_version not in MANIFEST_ENTRY_SCHEMAS
        ):
            raise ValueError(
                f"unsupported Iceberg metadata format version: {format_version}"
            )
        expected_list = self._manifest_list_ref(io, snapshot_id, snapshot.manifest_list)
        if cursor.state.manifest_list != expected_list:
            raise ValueError(
                "manifest-list identity, length, or digest changed during resume"
            )
        if cursor.state.manifest is None:
            return self._read_manifest_list(
                io, cursor.state, format_version, max_metadata_block_bytes
            )
        return self._read_manifest(
            io,
            cursor.state,
            format_version,
            max_records,
            max_units,
            max_metadata_block_bytes,
        )

    def _read_manifest_list(
        self,
        io: Any,
        state: TraversalState,
        format_version: int,
        block_limit: int,
    ) -> FileScanResult:
        input_file = io.new_input(state.manifest_list.path)
        if len(input_file) != state.manifest_list.object_length:
            raise ValueError("manifest-list object length changed during read")
        with input_file.open() as stream:
            reader = BoundedAvroBlocks(stream, max_block_bytes=block_limit)
            self._seek_list(reader, state)
            block = next(reader, None)
            if block is None:
                return self._result((), state, snapshot_exhausted=True)
            resolver = resolve_reader(
                reader.schema,
                MANIFEST_LIST_FILE_SCHEMAS[format_version],
                {-1: ManifestFile},
            )
            for row, manifest in enumerate(reader.iter_records(block, resolver)):
                if row < state.manifest_list_offset.row_offset:
                    continue
                ref = self._manifest_ref(io, manifest)
                if ref.content != str(ManifestContent.DATA.value):
                    raise ValueError("Iceberg delete manifest content is unsupported")
                next_state = replace(
                    state,
                    manifest_list_offset=ManifestListOffset(
                        block.start_offset, block.index, row
                    ),
                    manifest=ref,
                    manifest_entry_offset=ManifestEntryOffset(0, 0, 0),
                )
                return self._result((), next_state, snapshot_exhausted=False)
            next_state = replace(
                state,
                manifest_list_offset=ManifestListOffset(
                    block.next_offset, block.index + 1, 0
                ),
            )
            return self._result((), next_state, snapshot_exhausted=False)

    def _read_manifest(
        self,
        io: Any,
        state: TraversalState,
        format_version: int,
        max_records: int,
        max_units: int,
        block_limit: int,
    ) -> FileScanResult:
        assert state.manifest is not None and state.manifest_entry_offset is not None
        input_file = io.new_input(state.manifest.path)
        if len(input_file) != state.manifest.object_length:
            raise ValueError("manifest object length changed during read")
        with input_file.open() as stream:
            reader = BoundedAvroBlocks(stream, max_block_bytes=block_limit)
            self._seek_manifest(reader, state)
            block = next(reader, None)
            if block is None:
                offset = state.manifest_list_offset
                return self._result(
                    (),
                    state.after_manifest(
                        next_offset=ManifestListOffset(
                            offset.block_start,
                            offset.block_index,
                            offset.row_offset + 1,
                        )
                    ),
                    snapshot_exhausted=False,
                )
            resolver = resolve_reader(
                reader.schema,
                MANIFEST_ENTRY_SCHEMAS[format_version],
                {-1: ManifestEntry, 2: DataFile},
                {0: ManifestEntryStatus, 101: FileFormat, 134: DataFileContent},
            )
            manifest = ManifestFile.from_args(
                manifest_path=state.manifest.path,
                manifest_length=state.manifest.object_length,
                content=ManifestContent(int(state.manifest.content)),
                sequence_number=state.manifest.sequence,
                added_snapshot_id=state.manifest.added_snapshot_id,
                partition_spec_id=state.manifest.partition_spec_id,
            )
            files: list[FileScanRange] = []
            records = 0
            current = state
            for row, entry in enumerate(reader.iter_records(block, resolver)):
                if row < state.manifest_entry_offset.row_offset:
                    continue
                entry = _inherit_from_manifest(entry, manifest)
                after_entry = current.with_manifest_offset(
                    ManifestEntryOffset(block.start_offset, block.index, row + 1)
                )
                if state.additions_only and entry.snapshot_id != state.snapshot_id:
                    current = after_entry
                    continue
                if entry.status is ManifestEntryStatus.DELETED:
                    if state.additions_only:
                        raise ValueError("Iceberg append snapshot deletes data files")
                    current = after_entry
                    continue
                if entry.data_file.content is not DataFileContent.DATA:
                    raise ValueError("Iceberg manifest contains a non-data file")
                path, total_count, record_format = self._data_file_fact(entry.data_file)
                record_start = current.file_record_offset
                if record_start > total_count:
                    raise ValueError("file cursor exceeds metadata record count")
                if record_start == total_count:
                    current = after_entry
                    continue
                count = min(total_count - record_start, max_records - records)
                range_cursor = IcebergScanCursor(
                    replace(
                        current,
                        manifest_entry_offset=ManifestEntryOffset(
                            block.start_offset, block.index, row
                        ),
                    )
                )
                file_range = FileScanRange(path, count, record_format, range_cursor)
                files.append(file_range)
                records += count
                current = (
                    file_range.cursor_after(count).state
                    if record_start + count < total_count
                    else after_entry
                )
                if records == max_records or len(files) == max_units:
                    return self._result(tuple(files), current, snapshot_exhausted=False)
            current = current.with_manifest_offset(
                ManifestEntryOffset(block.next_offset, block.index + 1, 0)
            )
            return self._result(
                tuple(files),
                current,
                snapshot_exhausted=False,
            )

    @staticmethod
    def _result(
        files: tuple[FileScanRange, ...],
        state: TraversalState,
        *,
        snapshot_exhausted: bool,
    ) -> FileScanResult:
        return FileScanResult(
            files=files,
            next_cursor=IcebergScanCursor(state),
            snapshot_exhausted=snapshot_exhausted,
        )

    @staticmethod
    def _manifest_list_ref(io: Any, snapshot_id: int, path: str) -> ManifestListRef:
        input_file = io.new_input(path)
        return ManifestListRef.create(
            snapshot_id=snapshot_id, path=path, object_length=len(input_file)
        )

    @staticmethod
    def _manifest_ref(io: Any, manifest: ManifestFile) -> ManifestRef:
        path = getattr(manifest, "manifest_path", None)
        if not isinstance(path, str) or not path:
            raise ValueError("manifest list entry has no manifest path")
        declared_length = int(getattr(manifest, "manifest_length", 0))
        actual_length = len(io.new_input(path))
        if declared_length <= 0 or declared_length != actual_length:
            raise ValueError(
                "manifest object length does not match manifest-list entry"
            )
        raw_content = getattr(manifest, "content", "")
        content = str(getattr(raw_content, "value", raw_content))
        sequence = int(getattr(manifest, "sequence_number", 0))
        added_snapshot_id = manifest.added_snapshot_id
        if added_snapshot_id is None:
            raise ValueError("persisted Iceberg manifest lacks added_snapshot_id")
        return ManifestRef.create(
            path=path,
            object_length=actual_length,
            content=content,
            sequence=sequence,
            added_snapshot_id=added_snapshot_id,
            partition_spec_id=manifest.partition_spec_id,
        )

    @staticmethod
    def _data_file_fact(data_file: DataFile) -> tuple[str, int, str]:
        path = str(data_file.file_path)
        count = int(data_file.record_count)
        raw_format = getattr(data_file.file_format, "value", data_file.file_format)
        file_format = str(raw_format).lower()
        if not path or count <= 0 or not file_format:
            raise ValueError("Iceberg data file metadata is invalid")
        return path, count, file_format

    @staticmethod
    def _format_version(table: Any) -> int:
        value = table.metadata.format_version
        if type(value) is not int:
            raise TypeError("Iceberg table format version must be an integer")
        return value

    @staticmethod
    def _seek_list(reader: BoundedAvroBlocks, state: TraversalState) -> None:
        offset = state.manifest_list_offset
        if offset.block_index == 0 and offset.block_start == 0:
            return
        reader.seek(offset.block_start, block_index=offset.block_index)

    @staticmethod
    def _seek_manifest(reader: BoundedAvroBlocks, state: TraversalState) -> None:
        assert state.manifest_entry_offset is not None
        offset = state.manifest_entry_offset
        if offset.block_index == 0 and offset.block_start == 0:
            return
        reader.seek(offset.block_start, block_index=offset.block_index)


__all__ = ["IcebergManifestTraversal"]
