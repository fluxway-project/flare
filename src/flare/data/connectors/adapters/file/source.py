from __future__ import annotations

import hashlib
import json
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyarrow.fs import FileSelector, FileSystem, FileType

from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)
from flare.data.schema.feature_catalog import validate_storage_columns
from flare.io.aws import get_pyarrow_s3_fs

RangeReader = Callable[..., Any]
RecordCounter = Callable[[str], int]


def resolve_file_paths(
    dataset_path: str | Sequence[str],
    format_id: str,
    storage_options: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if isinstance(dataset_path, (list, tuple)):
        paths = tuple(p for p in dataset_path if p)
        if not paths:
            raise ValueError("dataset_path sequence must contain at least one path")
        return paths

    if (
        not isinstance(dataset_path, str)
        or not dataset_path
        or dataset_path.strip() != dataset_path
    ):
        raise ValueError("dataset_path must be a non-empty canonical string")
    if (
        not isinstance(format_id, str)
        or not format_id
        or format_id.strip() != format_id
    ):
        raise ValueError("format_id must be a non-empty canonical string")

    parsed = urllib.parse.urlparse(dataset_path)
    if parsed.scheme == "s3":
        fs = get_pyarrow_s3_fs(dict(storage_options or {}))
        if fs is None:
            raise ValueError("S3 file source storage_options are incomplete")
        canonical_dataset_path = f"s3://{parsed.netloc}{parsed.path}".rstrip("/")
        object_path = canonical_dataset_path[5:]
        object_info = fs.get_file_info(object_path)
        if object_info.type == FileType.File:
            selected_infos = [object_info]
        elif object_info.type == FileType.NotFound:
            raise FileNotFoundError(f"S3 path not found: {dataset_path}")
        else:
            selector = FileSelector(object_path, recursive=True)
            selected_infos = [
                info
                for info in fs.get_file_info(selector)
                if info.type == FileType.File and info.path.endswith(f".{format_id}")
            ]
        paths = tuple(sorted(f"s3://{info.path}" for info in selected_infos))
        if not paths:
            raise FileNotFoundError(
                f"no {format_id} files found in S3 path {dataset_path!r}"
            )
        return paths

    canonical_dataset_path = str(Path(dataset_path).expanduser().resolve())
    fs_obj, path = FileSystem.from_uri(canonical_dataset_path)
    path_info = fs_obj.get_file_info(path)
    if path_info.type == FileType.File:
        selected_infos = [path_info]
    elif path_info.type == FileType.NotFound:
        raise FileNotFoundError(f"path not found: {dataset_path}")
    else:
        selector = FileSelector(path, recursive=True)
        selected_infos = [
            info
            for info in fs_obj.get_file_info(selector)
            if info.type == FileType.File and info.path.endswith(f".{format_id}")
        ]
    paths = tuple(sorted(str(info.path) for info in selected_infos))
    if not paths:
        raise FileNotFoundError(f"no {format_id} files found in path {dataset_path!r}")
    return paths


class FileListSourceAdapter:
    def __init__(
        self,
        *,
        format_id: str,
        files: Sequence[str],
        projection: Sequence[str],
        range_reader: RangeReader,
        record_counter: RecordCounter,
    ) -> None:
        if (
            not isinstance(format_id, str)
            or not format_id
            or format_id.strip() != format_id
        ):
            raise ValueError("file source format_id must be a canonical string")
        if not files or any(not path for path in files):
            raise ValueError("files must contain at least one non-empty path")
        if not projection:
            raise ValueError("projection must contain at least one column")
        if not callable(range_reader):
            raise TypeError("file source range_reader must be callable")
        if not callable(record_counter):
            raise TypeError("file source record_counter must be callable")
        self._format_id = format_id
        self._files = tuple(files)
        self._projection = validate_storage_columns(projection)
        self._range_reader = range_reader
        self._record_counter = record_counter
        self.schema_digest = hashlib.sha256(
            json.dumps(self._projection, separators=(",", ":")).encode()
        ).hexdigest()
        revision_key = json.dumps(
            {
                "format": format_id,
                "files": self._files,
                "projection": self._projection,
            },
            separators=(",", ":"),
        ).encode()
        self._revision_identity = hashlib.sha256(revision_key).hexdigest()
        self.source_identity = f"{format_id}://{self._revision_identity}"

    @property
    def revision_identity(self) -> str:
        return self._revision_identity

    def bind_source_identity(self, source_identity: str) -> None:
        if (
            not isinstance(source_identity, str)
            or not source_identity
            or source_identity.strip() != source_identity
        ):
            raise ValueError("file source identity must be canonical")
        if source_identity != self.source_identity:
            raise ValueError("file source identity differs from assignment")

    def close(self) -> None:
        return None

    def record_ranges(self) -> tuple[SourceRecordRange, ...]:
        return tuple(
            SourceRecordRange(
                source_identity=self.source_identity,
                revision_identity=self._revision_identity,
                locator=FileLocator(path),
                record_start=0,
                record_count=self._count(path),
                schema_digest=self.schema_digest,
                record_format=self._format_id,
            )
            for path in self._files
        )

    def read(self, record_range: SourceRecordRange, projection: Sequence[str]) -> Any:
        if record_range.source_identity != self.source_identity:
            raise ValueError("record range belongs to another file-list source")
        if record_range.record_format != self._format_id:
            raise ValueError("record range format does not match file-list source")
        if tuple(projection) != self._projection:
            raise ValueError(
                "read projection does not match immutable source projection"
            )
        return self._range_reader(
            record_range.locator,
            projection=self._projection,
            row_start=record_range.record_start,
            row_count=record_range.record_count,
        )

    def read_batches(
        self,
        record_range: SourceRecordRange,
        projection: Sequence[str],
        *,
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("file source batch_size must be a positive integer")
        table = self.read(record_range, projection)
        if not isinstance(table, pa.Table):
            raise TypeError("file source range reader must return a pyarrow.Table")
        return iter(table.to_batches(max_chunksize=batch_size))

    def _count(self, path: str) -> int:
        count = self._record_counter(path)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"record counter returned invalid count for {path!r}")
        return count
