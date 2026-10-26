"""Standard assigned record-range execution; native SDK sessions bypass it."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

# pyrefly: ignore[missing-import]
import pyarrow as pa
from fluxway import profiler
from fluxway.common.ingestion import SourceWorkItem, source_work_global_valid_count
from fluxway.common.metadata import unpack_metadata

from flare.data.contracts.assets import AssetRef
from flare.data.pipeline.consumption.source.contracts.record_range import (
    RecordRangeReader,
)
from flare.data.pipeline.consumption.source.contracts.source import SourceBatch
from flare.data.pipeline.consumption.source.execution.invocation import invoke_source
from flare.data.pipeline.consumption.source.execution.materialization import (
    materialize_source_ranges,
    validate_source_read_concurrency,
)
from flare.data.schema.virtual_pad import (
    ArrowVirtualPadBatchFactory,
    VirtualPadBatchFactory,
)
from flare.data.sources.range import SourceRecordRange

_RANGE_CACHE_SIZE = 128


class RecordRangeReadSession:
    """Materialize exactly the SourceWorkItems acquired by Fluxway."""

    def __init__(
        self,
        *,
        reader: RecordRangeReader,
        projection: Sequence[str],
        asset: AssetRef | None = None,
        max_concurrent_reads: int = 2,
        pad_batch_factory: VirtualPadBatchFactory = ArrowVirtualPadBatchFactory.create_batch,
    ) -> None:
        if not isinstance(reader, RecordRangeReader):
            raise TypeError("assigned source reader requires RecordRangeReader")
        self._reader = reader
        self._asset = asset
        self._projection = tuple(projection)
        self._concurrency = validate_source_read_concurrency(max_concurrent_reads)
        if not callable(pad_batch_factory):
            raise TypeError("assigned source reader requires a pad batch factory")
        self._pad_batch_factory = pad_batch_factory
        self._closed = False
        self._bound_source_identity: str | None = None
        self._cached_schema: pa.Schema | None = None
        self._range_cache: OrderedDict[
            tuple[bytes, str, str, str, str], SourceRecordRange
        ] = OrderedDict()

    @property
    def schema(self) -> pa.Schema:
        if self._cached_schema is not None:
            return self._cached_schema
        reader_schema = getattr(self._reader, "schema", None)
        if not isinstance(reader_schema, pa.Schema):
            raise TypeError(
                "padding-only source reader requires an explicit Arrow schema"
            )
        self._cached_schema = reader_schema
        return reader_schema

    @property
    def bound_source_identity(self) -> str | None:
        return self._bound_source_identity

    def bind_source_identity(self, source_identity: str) -> None:
        if (
            self._bound_source_identity is not None
            and self._bound_source_identity != source_identity
        ):
            raise ValueError("assigned source identity changed during reader session")
        if self._bound_source_identity is None:
            self._reader.bind_source_identity(source_identity)
            self._bound_source_identity = source_identity

    async def read(
        self,
        step: int,
        *,
        work_items: Sequence[SourceWorkItem],
    ) -> SourceBatch:
        if self._closed:
            raise RuntimeError("assigned source reader is closed")
        real_items: list[SourceWorkItem] = []
        pad_items: list[SourceWorkItem] = []
        for item in work_items:
            self.bind_source_identity(item.assignment.source_identity)
            if item.is_synthetic_padding:
                pad_items.append(item)
            else:
                real_items.append(item)

        global_valid_count = source_work_global_valid_count(work_items)
        pad_rows = sum(item.cardinality for item in pad_items)
        table: pa.Table | None = None
        ranges: list[SourceRecordRange] = []
        if real_items:
            if profiler.enabled():
                with profiler.scope("sourcework.validate_assignments"):
                    ranges = self._assigned_ranges(real_items)
                with profiler.scope("sourcework.materialize_ranges"):
                    table = await self._materialize_ranges(ranges)
            else:
                ranges = self._assigned_ranges(real_items)
                table = await self._materialize_ranges(ranges)
            self._cached_schema = table.schema
            if pad_rows > 0:
                pad_table = pa.Table.from_batches(
                    [self._pad_batch_factory(table.schema, self._projection, pad_rows)]
                )
                table = pa.concat_tables([table, pad_table])
        else:
            schema = await invoke_source(lambda: self.schema)
            table = pa.Table.from_batches(
                [self._pad_batch_factory(schema, self._projection, pad_rows)]
            )

        if not isinstance(table, pa.Table):
            raise TypeError("record range materialization must return an Arrow table")

        existing_meta = dict(table.schema.metadata or {})
        existing_meta[b"samples_valid_count"] = str(global_valid_count).encode("utf-8")
        inputs = ()
        if self._asset is not None:
            inputs = tuple(
                AssetRef(self._asset.namespace, self._asset.name, revision)
                for revision in dict.fromkeys(item.revision_identity for item in ranges)
            )
        return SourceBatch(
            table.replace_schema_metadata(existing_meta), table.num_rows, inputs
        )

    def _assigned_ranges(
        self, work_items: Sequence[SourceWorkItem]
    ) -> list[SourceRecordRange]:
        ranges = []
        for item in work_items:
            assignment = item.assignment
            descriptor = assignment.descriptor
            opaque_token = descriptor.opaque_token
            if not isinstance(opaque_token, bytes):
                raise TypeError("source assignment opaque_token must be bytes")
            # Bind every compared identity into the key: a cached token must never
            # hide a conflicting assignment, including across planning windows.
            key = (
                opaque_token,
                assignment.source_identity,
                assignment.schema_digest,
                descriptor.revision_identity,
                descriptor.storage_locator,
            )
            record_range = self._range_cache.get(key)
            if record_range is None:
                record_range = SourceRecordRange.from_mapping(
                    unpack_metadata(opaque_token)
                )
                if (
                    record_range.source_identity != assignment.source_identity
                    or record_range.schema_digest != assignment.schema_digest
                    or record_range.revision_identity != descriptor.revision_identity
                    or record_range.locator.canonical() != descriptor.storage_locator
                ):
                    raise ValueError("source range metadata conflicts with assignment")
                self._range_cache[key] = record_range
                if len(self._range_cache) > _RANGE_CACHE_SIZE:
                    self._range_cache.popitem(last=False)
            else:
                self._range_cache.move_to_end(key)
            ranges.append(
                record_range.slice(
                    item.logical_start - record_range.record_start, item.cardinality
                )
            )
        return ranges

    async def _materialize_ranges(
        self, ranges: Sequence[SourceRecordRange]
    ) -> pa.Table:
        return await materialize_source_ranges(
            ranges,
            read=self._reader.read,
            projection=self._projection,
            max_concurrent_reads=self._concurrency,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._range_cache.clear()
        await invoke_source(self._reader.close)
