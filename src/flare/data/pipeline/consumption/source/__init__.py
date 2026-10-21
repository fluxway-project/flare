"""Source-neutral streaming data contracts and batch assembly."""

from flare.data.pipeline.consumption.source.batching import (
    BatchAssembler,
    BatchCompletionPolicy,
    BatchValidity,
    CarryUntilComplete,
    DropIncompleteAtEnd,
    PadWithValidity,
    RequireComplete,
)
from flare.data.pipeline.consumption.source.batching.stream_reader import (
    StreamingBatchReader,
)
from flare.data.pipeline.consumption.source.contracts import (
    ControlledSourceScanAdapter,
    SourceFactory,
    SourceReader,
    SourceRecordRange,
)
from flare.data.pipeline.consumption.source.placement import (
    bind_partitions,
    stable_owner,
)

__all__ = [
    "BatchAssembler",
    "BatchCompletionPolicy",
    "BatchValidity",
    "CarryUntilComplete",
    "DropIncompleteAtEnd",
    "PadWithValidity",
    "RequireComplete",
    "SourceFactory",
    "SourceReader",
    "ControlledSourceScanAdapter",
    "SourceRecordRange",
    "StreamingBatchReader",
    "bind_partitions",
    "stable_owner",
]
