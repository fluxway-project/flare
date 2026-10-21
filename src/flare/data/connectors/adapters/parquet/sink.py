from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from flare.data.connectors.adapters.iceberg.table import (
    PARQUET_COMPRESSION,
    PARQUET_VERSION,
)
from flare.data.contracts.sink import SinkResult
from flare.data.pipeline.production.publication.contract import PublicationProtocol
from flare.data.pipeline.production.publish.partitioning import (
    parquet_partition_directory,
    require_partition_fields,
)
from flare.data.schema.feature_catalog import validate_storage_columns


class PyArrowBatchSource(Protocol):
    def iter_batches(
        self,
        *,
        batch_size: int | None,
        batch_format: str,
        prefetch_batches: int,
    ) -> Iterator[pa.Table]: ...


class ParquetDatasetPublisher:
    @property
    def publication_protocol(self) -> PublicationProtocol:
        return PublicationProtocol.NON_TRANSACTIONAL

    def __init__(
        self,
        root: str | Path,
        *,
        prefetch_batches: int = 1,
        max_rows_per_file: int = 65_536,
        overwrite: bool = False,
    ) -> None:
        self._root = Path(root).resolve()
        if self._root == Path(self._root.anchor):
            raise ValueError("Parquet output root must not be a filesystem root")
        if self._root.exists():
            if not overwrite:
                raise FileExistsError(self._root)
            if self._root.is_dir():
                shutil.rmtree(self._root)
            else:
                self._root.unlink()
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative")
        if max_rows_per_file <= 0:
            raise ValueError("max_rows_per_file must be positive")
        self._root.mkdir(parents=True, exist_ok=False)
        self._prefetch_batches = prefetch_batches
        self._max_rows_per_file = max_rows_per_file
        self._next_file_index = 0

    @property
    def root(self) -> Path:
        return self._root

    def finalize(
        self,
        ds: PyArrowBatchSource,
        *,
        schema: pa.Schema,
    ) -> SinkResult:
        if not isinstance(schema, pa.Schema):
            raise TypeError("ParquetDatasetPublisher requires a PyArrow schema")
        validate_storage_columns(schema.names)
        partition_schema = pa.schema(require_partition_fields(schema))
        data_schema = pa.schema(
            field for field in schema if field.name not in partition_schema.names
        )
        total_rows = 0
        writer: pq.ParquetWriter | None = None
        writer_partition: tuple[str, str] | None = None
        writer_rows = 0
        file_index = self._next_file_index
        try:
            for table in ds.iter_batches(
                batch_size=None,
                batch_format="pyarrow",
                prefetch_batches=self._prefetch_batches,
            ):
                if not isinstance(table, pa.Table):
                    raise TypeError(
                        "ParquetDatasetPublisher requires PyArrow Table batches"
                    )
                if not table.schema.equals(schema, check_metadata=False):
                    expected_fields = {field.name: field for field in schema}
                    actual_fields = {field.name: field for field in table.schema}
                    missing = sorted(set(expected_fields) - set(actual_fields))
                    extra = sorted(set(actual_fields) - set(expected_fields))
                    type_mismatches = {
                        name: (
                            str(expected_fields[name].type),
                            str(actual_fields[name].type),
                        )
                        for name in expected_fields.keys() & actual_fields.keys()
                        if expected_fields[name].type != actual_fields[name].type
                    }
                    raise ValueError(
                        "Parquet output batch schema changed during execution: "
                        f"missing={missing} extra={extra} "
                        f"type_mismatches={type_mismatches} "
                        f"order_matches={table.column_names == schema.names}"
                    )
                partition = (
                    self._partition_value(table, partition_schema.names[0]),
                    self._partition_value(table, partition_schema.names[1]),
                )
                data = table.drop(partition_schema.names)
                offset = 0
                while offset < data.num_rows:
                    if writer is None or writer_partition != partition:
                        if writer is not None:
                            writer.close()
                        writer = self._open_writer(data_schema, partition, file_index)
                        writer_partition = partition
                        writer_rows = 0
                        file_index += 1
                    take = min(
                        self._max_rows_per_file - writer_rows, data.num_rows - offset
                    )
                    writer.write_table(data.slice(offset, take), row_group_size=take)
                    offset += take
                    writer_rows += take
                    if writer_rows == self._max_rows_per_file:
                        writer.close()
                        writer = None
                total_rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        if total_rows == 0:
            raise ValueError("Parquet output dataset must not be empty")
        self._next_file_index = file_index
        return SinkResult(total_rows=total_rows)

    @staticmethod
    def _partition_value(table: pa.Table, name: str) -> str:
        values = pc.call_function("unique", [table[name].combine_chunks()])
        if len(values) != 1 or values[0].as_py() is None:
            raise ValueError(f"Parquet output batch must contain one {name} value")
        value = values[0].as_py()
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(f"invalid Parquet partition value {name}={value!r}")
        return value

    def _open_writer(
        self, schema: pa.Schema, partition: tuple[str, str], file_index: int
    ) -> pq.ParquetWriter:
        directory = parquet_partition_directory(self._root, partition)
        directory.mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(
            directory / f"part-{file_index:08d}.parquet",
            schema,
            compression=PARQUET_COMPRESSION,
            version=PARQUET_VERSION,
            write_statistics=False,
            use_dictionary=False,
        )
