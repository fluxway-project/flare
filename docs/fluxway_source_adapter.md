# Flare and Fluxway Source Adapter

This document specifies the public source adapter contracts and runtime interfaces between Flare and Fluxway.

## Module Responsibilities

| Entrypoint | Responsibility |
| --- | --- |
| `src/flare/data/connectors/adapters/iceberg/source.py` | Native snapshot discovery, file intervals, and consumption locations |
| `src/flare/data/connectors/adapters/iceberg/scan.py` | Converts native file intervals and cursors into typed scan results |
| `src/flare/runtime/fluxway/source.py` | Source binding, reader topology and session construction |
| `src/flare/data/pipeline/consumption/source/execution/record_ranges.py` | Standard file-range validation and materialization |
| `src/flare/runtime/fluxway/workers/reader.py` | Generic reader worker, task activation, and checkpoint validation |
| `src/flare/data/connectors/adapters/parquet/range_reader.py` | Projected range reads, file caching and shutdown |
| `src/flare/data/pipeline/consumption/dataloader` | Standalone file loading, bounded reads and buffer utilities |

The source boundary uses separate scan-planner and physical-reader interfaces.
`ControlledSourceScanAdapter` provides an opaque initial cursor,
`scan_next(cursor, budget)` and non-I/O `advance_cursor(cursor, records)`.
Scans return typed, ordered source units and cursor metadata; the planner owns
format-specific revision state. `SourceReadSession.read(step, work_items=...)`
returns `SourceBatch(payload, cardinality, inputs)`. Each source interprets its
own opaque tasks; payloads retain their native representation. Factories own
construction and configuration validation for both interfaces. Both ports
require `close()` and support synchronous or asynchronous implementations.

See [dataloader execution](dataloader.md) for construction and physical read
sequences, and [streaming sources](streaming_source.md) for append and recovery
sequences.

`build_reader_source()` composes a `WorkItemScheduledDatasetSourceDescriptor`
with projection, read concurrency and format-specific options, or accepts a
direct Fluxway read session. These two modes are exclusive. The assigned mode
lets the compiled Fluxway graph derive topology and work assignments:

```python
from flare.runtime.fluxway.workers.reader_contracts import build_reader_source
from flare.runtime.fluxway.source import WorkItemScheduledDatasetSourceDescriptor

source = build_reader_source(
    descriptor=WorkItemScheduledDatasetSourceDescriptor(
        source_factory=source_factory,
        global_batch_size=1024,
        completion_policy="pad",
    ),
    projection=("feature", "label", "group_id"),
    source_read_concurrency=4,
    storage_options={
        "storage_format": "parquet",
        "dataset_path": "/data/dataset",
    },
)
```

`source_factory` must implement `SourceFactory.prepare_options()`, scan-planner
creation and session creation. Projection and concurrency are optional at the
generic builder boundary; each factory validates its own required options.
Iceberg options are provided through `storage_options`;
catalog paths and individual catalog fields have an exclusive contract.

`RecordRangeReadSession` adapts acquired `SourceWorkItem` instances into
`SourceRecordRange`. It verifies step, source, schema, revision and physical
identity before materializing each real range. Synthetic padding work does not
contain a physical file range. Padding-only reads require the source's explicit
Arrow schema and perform no data-file read. Reader workers do not independently
discover new work, assign files or manage a second durable source cursor.

Native query or shard sessions implement `SourceReadSession` directly and use
their own task tokens and SDK execution. They do not pass through file-range
decoding. The default training worker consumes Arrow tables, record batches or
column mappings; other representations belong to a consumer implementation of
`transform_raw_batch()`. The source interface does not promise replay or
parallel partitioning that the backend cannot provide.

Iceberg catalog construction is injected through `IcebergCatalogProvider`.
`IcebergTableSession` owns its catalog context, table UUID and refresh checks;
format readers reuse the table's authorized FileIO. File access, publication
and external lineage are independent capabilities. Optional `LineageConfig`
creates one `LineageRecorder` with bounded metadata aggregation and a batch
`LineageExporter`. Reads record confirmed input assets after success; output
facts follow confirmed publication. The descriptor passes
`include_lineage=self.lineage is not None` to `SourceFactory.create_reader()`;
standard sessions skip input asset construction when it is false.
Unconfigured lineage creates no exporter or background task. Lineage delivery
does not commit data or publish source checkpoint state.

`LineageRecorder(config)` only assembles configuration. The first nonempty
flush opens the exporter inside the recorder's operation owner. Cancellation
of a caller does not abandon creation, reads or delivery: close stops admission,
waits for accepted operations, then releases resources. Calling close from an
operation that it would wait for raises an error, including descendant tasks.
Native network timeouts belong to each backend. A caller's shutdown deadline
does not stop a synchronous SDK call that is already running.

## Windows and Batch Validity

- **Window Lifecycle**: A source window bounds admission and prefetch rather than discarding data. Acquired but unconsumed `SourceWorkItem` values remain with their owner for the next contiguous activation; the ledger owns leases and acknowledgements. Readers wait asynchronously for unpublished steps or handle typed `SOURCE_EXHAUSTED`.
- **Sample Validity**: Assigned Reader output carries the global valid-prefix
  count for every batch, including a complete batch. Consumers preserve this
  metadata through their payloads. A terminal partial batch is trainable
  only when the `pad` policy emits fixed-shape tensors; losses, sparse lookup
  preparation and metrics mask rows beyond that count. Reader descriptors map
  `carry`, `drop`, `pad` and `require` to retain-pending, discard, synthesized
  padding and reject-remainder layout modes respectively.

## Checkpoint and Recovery

Fluxway source-control components persist identity, consumption cursor and the
coordinated cut. That cursor is derived at the cut and can lag scan progress.
Reader workers register their `StateCheckpointComponent` via
`source_checkpoint_registration()`, validating the Reader position against the
coordinated cut without persisting source cursors, physical file offsets or
prefetched payloads.

Recovery uses `RestoreTopologyMode.GLOBAL_RESHARD`: it restores logical positions and replans work distribution across current reader workers, discarding transient iterators and prefetched buffers.
