# Flare and Fluxway Source Adapter

This document specifies the public source adapter contracts and runtime interfaces between Flare and Fluxway.

## Module Responsibilities

| Entrypoint | Responsibility |
| --- | --- |
| `src/flare/data/connectors/adapters/iceberg/source.py` | Native snapshot discovery, file intervals, and consumption locations |
| `src/flare/runtime/fluxway/source.py` | Translates file intervals and positions into scan results and work items |
| `src/flare/runtime/fluxway/workers/reader.py` | Generic reader worker, task activation, and checkpoint validation |
| `src/flare/data/pipeline/consumption/dataloader` | File decoding, record batch buffering, and device delivery |

The source boundary uses separate scan-planner and physical-reader interfaces.
`ControlledSourceScanAdapter` provides an opaque initial cursor,
`scan_next(cursor, budget)` and non-I/O `advance_cursor(cursor, records)`.
Scans return typed, ordered source units and cursor metadata; the planner owns
format-specific revision state. `SourceReader` provides assigned-range reads
and batch reads. Factories own construction of both interfaces.

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

`source_factory` must implement `SourceFactory`, including both scan-planner
and Reader creation. Iceberg options are provided through `storage_options`;
catalog paths and individual catalog fields have an exclusive contract.

`AssignedSourceBatchReader` adapts acquired `SourceWorkItem` instances into
`SourceRecordRange`. It verifies step, source, schema, revision and physical
identity before materializing each real range. Synthetic padding work does not
contain a physical file range. Padding-only reads require the source's explicit
Arrow schema and perform no data read. Reader workers do not poll catalogs,
assign files or manage independent durable cursors.

## Windows and Batch Validity

- **Window Lifecycle**: A source window is an admission and prefetch boundary, not a discard boundary. Unconsumed `SourceWorkItem` items remain in canonical pending state and are scheduled into the next contiguous activation exactly once. Readers wait asynchronously for published activations or handle typed `SOURCE_EXHAUSTED`.
- **Sample Validity**: Assigned Reader output carries the global valid-prefix
  count for every batch, including a complete batch. Consumers preserve this
  metadata through their payloads. A terminal partial batch is trainable
  only when the `pad` policy emits fixed-shape tensors; losses, sparse lookup
  preparation and metrics mask rows beyond that count. Reader descriptors map
  `carry`, `drop`, `pad` and `require` to retain-pending, discard, synthesized
  padding and reject-remainder layout modes respectively.

## Checkpoint and Recovery

Fluxway source-control components persist the scan state and coordinated cut.
Reader workers register their `StateCheckpointComponent` via
`source_checkpoint_registration()`, validating the Reader position against the
coordinated cut without persisting source cursors, physical file offsets or
prefetched payloads.

Recovery uses `RestoreTopologyMode.GLOBAL_RESHARD`: it restores logical positions and replans work distribution across current reader workers, discarding transient iterators and prefetched buffers.
