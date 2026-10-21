from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flare.data.pipeline.production.builder.constants import (
    DEFAULT_FLOAT_NULL_SENTINEL,
    DEFAULT_MISSING_FLOAT,
    DEFAULT_MISSING_INT,
    DEFAULT_NULL_SENTINEL,
    DEFAULT_PAD_VALUE,
)


@dataclass(frozen=True, slots=True)
class SentinelConfig:
    null_sentinel: int = DEFAULT_NULL_SENTINEL
    float_null_sentinel: float = DEFAULT_FLOAT_NULL_SENTINEL
    missing_int: int = DEFAULT_MISSING_INT
    missing_float: float = DEFAULT_MISSING_FLOAT
    pad_value: int = DEFAULT_PAD_VALUE


@dataclass(frozen=True, slots=True)
class StageConfig:
    source_block_size_bytes: int = 64 * 1024 * 1024
    target_block_size_bytes: int = 128 * 1024 * 1024
    read_concurrency: int | None = None
    parse_concurrency: int | None = None
    encode_concurrency: int | None = None
    write_concurrency: int = 1
    object_store_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class DataPipelineSpec:
    source: Any
    parser: Any
    transforms: Sequence[Any] = field(default_factory=tuple)
    sink: Any = None
    sentinels: SentinelConfig = field(default_factory=SentinelConfig)
    stage_config: StageConfig = field(default_factory=StageConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("DataPipelineSpec source must not be None")
        if self.parser is None:
            raise ValueError("DataPipelineSpec parser must not be None")


@runtime_checkable
class DataPipelineBuilder(Protocol):
    def build_pipeline_spec(
        self,
        source: Any,
        sink: Any = None,
        stage_config: StageConfig | None = None,
    ) -> DataPipelineSpec: ...


__all__ = [
    "DataPipelineBuilder",
    "DataPipelineSpec",
    "SentinelConfig",
    "StageConfig",
]
