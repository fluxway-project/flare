from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pyarrow as pa
from fluxway import profiler
from fluxway.common.decorators import requires_worker_process
from fluxway.common.ingestion import SourceWorkItem, source_work_global_valid_count
from fluxway.common.operator_policy import AsyncPolicy, StatePolicy
from fluxway.common.tensor_components import row_partition_sparse_feature
from fluxway.common.types.tensor import TransportTensor

from flare.data.schema.batch import RaggedFeatureBatch
from flare.data.schema.feature_layout import TrainingFeatureLayout
from flare.runtime.fluxway.source import (
    WorkItemScheduledDatasetSourceDescriptor,
)
from flare.runtime.fluxway.transport import transport_rebind, wrap_ragged_feature_batch
from flare.runtime.fluxway.workers.reader import (
    BatchReaderSPMD,
    source_checkpoint_registration,
)
from flare.runtime.fluxway.workers.reader_contracts import (
    ReaderSource,
)

logger = logging.getLogger(__name__)

BatchTransformer = Callable[[pa.Table], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class RankingReaderConfig:
    source: ReaderSource
    feature_layout: TrainingFeatureLayout
    sparse_groups: frozenset[str]
    transform: BatchTransformer
    output_type: type
    group_id_column: str = "group_id"
    repeat_first_batch: bool = False
    execution_audit_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, ReaderSource):
            raise TypeError("ranking reader requires a compiled reader source")
        if not isinstance(self.feature_layout, TrainingFeatureLayout):
            raise TypeError("ranking reader requires a TrainingFeatureLayout")
        if not isinstance(self.sparse_groups, frozenset):
            raise TypeError("ranking sparse_groups must be a frozenset")
        if not callable(self.transform):
            raise TypeError("ranking reader requires a callable transform")
        if not isinstance(self.output_type, type) or not hasattr(
            self.output_type, "__dataclass_fields__"
        ):
            raise TypeError("ranking reader requires a dataclass output_type")
        output_fields = set(self.output_type.__dataclass_fields__)
        missing = {"group_id", "samples_valid_count"}.difference(output_fields)
        if missing:
            raise ValueError(
                "ranking reader output_type is missing required fields: "
                f"{sorted(missing)!r}"
            )
        if not isinstance(self.group_id_column, str) or not self.group_id_column:
            raise ValueError("ranking group_id_column must be a non-empty string")
        if not isinstance(self.repeat_first_batch, bool):
            raise TypeError("ranking repeat_first_batch must be a boolean")


def resolve_reader_async_policy(reader_pipeline_depth: int) -> AsyncPolicy:
    if (
        not isinstance(reader_pipeline_depth, int)
        or isinstance(reader_pipeline_depth, bool)
        or reader_pipeline_depth <= 0
    ):
        raise ValueError("reader_pipeline_depth must be a positive integer")
    if reader_pipeline_depth == 1:
        return AsyncPolicy.sync()
    return AsyncPolicy.pipelined(
        depth=reader_pipeline_depth,
        state=StatePolicy.exclusive(),
    )


@requires_worker_process
class RecsysReaderSPMD(BatchReaderSPMD):
    rank: int
    world_size: int
    ctx: Any

    @property
    def feature_layout(self) -> TrainingFeatureLayout:
        return self._feature_layout

    def __init__(
        self,
        *,
        reader_config: RankingReaderConfig,
        name: str = "",
        mesh: Any = None,
        runtime_env: Any = None,
        resource: Any = None,
    ) -> None:
        if not isinstance(reader_config, RankingReaderConfig):
            raise TypeError("RecsysReaderSPMD requires RankingReaderConfig")
        super().__init__(
            source=replace(
                reader_config.source,
                pad_batch_factory=reader_config.feature_layout.create_virtual_pad_batch,
            ),
            output_builder=self._build_output,
            repeat_first_batch=reader_config.repeat_first_batch,
            execution_audit_path=reader_config.execution_audit_path,
            name=name,
            mesh=mesh,
            runtime_env=runtime_env,
            resource=resource,
        )
        self.sparse_groups = reader_config.sparse_groups
        self._feature_layout = reader_config.feature_layout
        self.group_id_column = reader_config.group_id_column
        self._batch_transformer = reader_config.transform
        self._output_type = reader_config.output_type

    def _build_output(self, tensors: Mapping[str, TransportTensor]) -> Any:
        field_names = tuple(self._output_type.__dataclass_fields__)
        missing = set(field_names).difference(tensors)
        if missing:
            raise ValueError(
                f"ranking reader output is missing fields: {sorted(missing)!r}"
            )
        return self._output_type(**{name: tensors[name] for name in field_names})

    @property
    def scheduled_global_valid_count(self) -> int | None:
        """Extract authoritative global valid count from scheduled work items."""
        context = getattr(self, "ctx", None)
        work_items: Sequence[SourceWorkItem] | None = getattr(
            context, "source_work_items", None
        )
        if not work_items:
            return None
        return source_work_global_valid_count(work_items)

    def _resolve_global_valid_count(
        self,
        raw_val: Any,
    ) -> int:
        """Resolve the canonical global valid-prefix count scalar for this batch."""
        scheduled = self.scheduled_global_valid_count
        if raw_val is not None:
            valid_count = self._valid_count_scalar(raw_val)
            if scheduled is not None and valid_count != scheduled:
                raise ValueError(
                    "samples_valid_count conflicts with scheduled source work"
                )
            return valid_count
        if scheduled is not None:
            return scheduled
        raise ValueError(
            "samples_valid_count requires explicit metadata or scheduled source work"
        )

    @staticmethod
    def _valid_count_scalar(raw_val: Any) -> int:
        arr = np.asarray(raw_val)
        if arr.dtype != np.dtype(np.int32) or arr.shape not in ((), (1,)):
            raise ValueError("samples_valid_count must be one int32 scalar")
        return int(arr.item())

    def _to_transport_batch(
        self, batch: Mapping[str, Any], logical_rank: int, logical_world: int
    ) -> Any:
        payload = dict(batch)
        if self.group_id_column not in payload:
            raise ValueError(
                f"training batch is missing group id column {self.group_id_column!r}"
            )
        group_id_val = payload.pop(self.group_id_column)
        valid_count_val = payload.pop("samples_valid_count", None)
        converted = {}

        bs = len(group_id_val)
        origins, global_rows = self._source_batch_coordinates(bs)
        group_id_arr = np.asarray(group_id_val, dtype=np.int64)
        converted["group_id"] = TransportTensor.from_tensor(
            group_id_arr,
            global_shape=(global_rows,),
            origin_indices=origins,
        )
        valid_count = self._resolve_global_valid_count(valid_count_val)

        if not 0 <= valid_count <= min(global_rows, np.iinfo(np.int32).max):
            raise ValueError(
                "samples_valid_count must be one integer within the global batch: "
                f"value={valid_count} global_batch_rows={global_rows}"
            )
        converted["samples_valid_count"] = TransportTensor.from_tensor(
            np.asarray([valid_count], dtype=np.int32),
            global_shape=(1,),
        )

        for name in self._output_type.__dataclass_fields__:
            if name in ("group_id", "samples_valid_count"):
                continue
            if name not in payload:
                raise ValueError(
                    f"training batch is missing ranking reader output field {name!r}"
                )
            value = payload[name]
            if isinstance(value, RaggedFeatureBatch):
                if value.row_count != bs:
                    raise ValueError(
                        f"sparse feature {name!r} row count differs from source batch"
                    )
                converted[name] = wrap_ragged_feature_batch(
                    value,
                    global_rows=global_rows,
                    origin_indices=origins,
                    source_step=self.ctx.step,
                )
                continue
            if isinstance(value, TransportTensor):
                g_shape = (
                    (global_rows, *value.global_shape[1:])
                    if value.global_shape is not None
                    else None
                )
                converted[name] = transport_rebind(
                    value, origin_indices=origins, global_shape=g_shape
                )
                continue

            if name in self.sparse_groups:
                sparse_layout = row_partition_sparse_feature(
                    name,
                    value,
                    dtype=np.int64,
                )
                g_shape = (
                    global_rows,
                    *sparse_layout.widths,
                )
                if sparse_layout.rows != bs:
                    raise ValueError(
                        "sparse feature row count differs from source batch"
                    )
                converted[name] = TransportTensor(
                    flat_values=sparse_layout.flat_values,
                    nested_splits=list(sparse_layout.nested_splits),
                    origin_indices=origins,
                    global_shape=g_shape,
                )
            else:
                array = np.asarray(value)
                if array.dtype == object:
                    raise TypeError(f"training batch field {name!r} has object dtype")
                g_shape = (global_rows, *array.shape[1:])
                converted[name] = TransportTensor.from_tensor(
                    array,
                    global_shape=g_shape,
                    origin_indices=origins,
                )

        return converted

    def transform_raw_batch(self, raw_batch: Any) -> Mapping[str, Any]:
        if isinstance(raw_batch, pa.RecordBatch):
            raw_batch = pa.Table.from_batches([raw_batch])
        if isinstance(raw_batch, pa.Table):
            with profiler.scope("reader.transform"):
                batch = self._batch_transformer(raw_batch)
            if not isinstance(batch, Mapping):
                raise TypeError("ranking batch transform must return a mapping")
            batch = dict(batch)
            metadata = raw_batch.schema.metadata or {}
            if b"samples_valid_count" in metadata:
                encoded = metadata[b"samples_valid_count"]
                valid_count = int(encoded)
                if (
                    encoded != str(valid_count).encode("ascii")
                    or not 0 <= valid_count <= np.iinfo(np.int32).max
                ):
                    raise ValueError(
                        "samples_valid_count metadata must be a canonical int32 count"
                    )
                if (
                    "samples_valid_count" in batch
                    and self._valid_count_scalar(batch["samples_valid_count"])
                    != valid_count
                ):
                    raise ValueError(
                        "samples_valid_count metadata conflicts with transformed batch"
                    )
                batch["samples_valid_count"] = np.asarray([valid_count], dtype=np.int32)
            return batch
        if isinstance(raw_batch, Mapping):
            return raw_batch
        raise TypeError(
            f"Reader batch must be an Arrow table or mapping, got {type(raw_batch).__name__}"
        )


def get_recsys_reader_class(
    sparse_groups: tuple[str, ...],
    runtime_env: dict[str, Any] | None,
    resource_profile: Any,
    reader_workers: int,
    async_policy: AsyncPolicy,
    source_descriptor: WorkItemScheduledDatasetSourceDescriptor | None = None,
    interleave_unit_size: int = 1,
    execution_groups: tuple[tuple[int, ...], ...] | None = None,
    batch_fields: tuple[str, ...] = (
        "side_info",
        "labels",
        "group_id",
        "samples_valid_count",
    ),
    heartbeat_timeout_s: float = 300.0,
    heartbeat_interval_s: float = 5.0,
) -> type:
    # pyrefly: ignore[missing-import]
    from fluxway.common.decorators import operator, spmd_source

    if not isinstance(async_policy, AsyncPolicy):
        raise TypeError("Reader async_policy must be AsyncPolicy")

    output_sharding = {}
    for g in sparse_groups:
        output_sharding[g] = {"type": "static", "axis": ["worker"]}
    for f in batch_fields:
        if f == "samples_valid_count":
            output_sharding[f] = {"type": "static", "axis": []}
        else:
            output_sharding[f] = {"type": "static", "axis": ["worker"]}

    source_class = type(
        RecsysReaderSPMD.__name__,
        (RecsysReaderSPMD,),
        {
            "__module__": RecsysReaderSPMD.__module__,
            "read": operator(
                name="read",
                async_policy=async_policy,
                output_sharding=output_sharding,
            )(RecsysReaderSPMD.read),
        },
    )
    return spmd_source(
        resource_profile=resource_profile,
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(reader_workers,),
        runtime_env=runtime_env,
        checkpoint_components=(source_checkpoint_registration(),),
        interleaved=True,
        interleave_unit_size=interleave_unit_size,
        execution_groups=execution_groups,
        source=source_descriptor,
        heartbeat_timeout_s=heartbeat_timeout_s,
        heartbeat_interval_s=heartbeat_interval_s,
    )(source_class)


__all__ = [
    "RankingReaderConfig",
    "RecsysReaderSPMD",
    "get_recsys_reader_class",
    "resolve_reader_async_policy",
]
