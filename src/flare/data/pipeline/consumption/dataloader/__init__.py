"""Framework-neutral asynchronous dataloader execution primitives."""

from flare.data.pipeline.consumption.dataloader.buffer import (
    BufferBackend,
    BufferCapabilities,
    ImmutableCpuBuffer,
    LeasedCpuBuffer,
)
from flare.data.pipeline.consumption.dataloader.device import DeviceTransferStage
from flare.data.pipeline.consumption.dataloader.executor import (
    AsyncBatchExecutor,
    AsyncReadSession,
    FormatAdapterRegistry,
    ReadLimits,
)
from flare.data.pipeline.consumption.dataloader.layout import (
    BatchLayoutSpec,
    DecodedBatch,
    DecodedField,
    FieldLayout,
)
from flare.data.pipeline.consumption.dataloader.parquet import (
    ParquetRangeBatchLoader,
)

__all__ = [
    "AsyncBatchExecutor",
    "AsyncReadSession",
    "BatchLayoutSpec",
    "BufferBackend",
    "BufferCapabilities",
    "DecodedBatch",
    "DecodedField",
    "DeviceTransferStage",
    "FieldLayout",
    "FormatAdapterRegistry",
    "ImmutableCpuBuffer",
    "LeasedCpuBuffer",
    "ParquetRangeBatchLoader",
    "ReadLimits",
]
