from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from flare.data.schema.batch import RaggedFeatureBatch
from flare.runtime.fluxway.source import (
    AssignedSourceBatchReader,
    WorkItemScheduledDatasetSourceDescriptor,
)
from flare.runtime.fluxway.transport import wrap_ragged_feature_batch
from flare.runtime.fluxway.workers.reader_contracts import ReaderSource
from fluxway import profiler
from fluxway.common.decorators import requires_worker_process
from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.types.checkpoint import (
    RestoreSession,
    RestoreTopologyMode,
    SaveSession,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.control_plane.checkpointer import StateCheckpointComponent
from fluxway.control_plane.ingestion import (
    IngestionLayoutActivation,
)
from fluxway.reader import BaseReader
from fluxway.reader.ingestion_layout import IngestionLayoutSession
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)

logger = logging.getLogger(__name__)


def _clone_transport_mapping(
    batch: Mapping[str, TransportTensor],
) -> dict[str, TransportTensor]:
    result: dict[str, TransportTensor] = {}
    for name, tensor in batch.items():
        if not isinstance(tensor, TransportTensor):
            raise TypeError(
                f"cached Reader field {name!r} must be TransportTensor, "
                f"got {type(tensor).__name__}"
            )
        if tensor.source_step is not None:
            raise ValueError(
                f"cached Reader template field {name!r} already belongs to "
                f"source_step={tensor.source_step}"
            )
        result[name] = TransportTensor(
            flat_values=tensor.flat_values,
            nested_splits=tensor.nested_splits,
            origin_indices=tensor.origin_indices,
            routing_indices=tensor.routing_indices,
            global_shape=tensor.global_shape,
            receipt=tensor.receipt,
            routing_axes=tensor.routing_axes,
        )
    return result


@requires_worker_process
class BatchReaderSPMD(BaseReader):
    """Generic SPMD batch reader worker decoupled from any specific task domain."""

    rank: int
    world_size: int
    ctx: Any

    def __init__(
        self,
        *,
        source: ReaderSource,
        output_builder: Any,
        repeat_first_batch: bool = False,
        execution_audit_path: str | None = None,
        name: str = "",
        mesh: Any = None,
        runtime_env: Any = None,
        resource: Any = None,
    ) -> None:
        super().__init__(
            name=name,
            mesh=mesh,
            runtime_env=runtime_env,
            resource=resource,
        )
        if not isinstance(source, ReaderSource):
            raise TypeError("BatchReaderSPMD requires a ReaderSource")
        if not callable(output_builder):
            raise TypeError("BatchReaderSPMD requires a callable output_builder")
        if not isinstance(repeat_first_batch, bool):
            raise TypeError("Reader repeat_first_batch must be a boolean")
        if execution_audit_path is not None and not isinstance(
            execution_audit_path, str
        ):
            raise TypeError("Reader execution_audit_path must be a string")
        self._source = source
        self._output_builder = output_builder
        self.repeat_first_batch = repeat_first_batch
        if execution_audit_path is not None:
            self._execution_audit_path = execution_audit_path
            self._execution_audit_records: list[dict[str, Any]] = []
            self._execution_audit_written = False
        self._cached_batch: Mapping[str, TransportTensor] | None = None
        if source.descriptor is not None:
            self._source_descriptor = source.descriptor
            self._assigned_source_options = dict(source.options)
            self._ingestion_read_session = None
            self._read_chunk_size = None
        else:
            self._source_descriptor = None
            self._assigned_source_options = {}
            self._ingestion_read_session = source.read_session
            self._read_chunk_size = source.read_chunk_size
        self._layout_sessions: dict[int, IngestionLayoutSession] = {}
        self._assigned_session: AssignedSourceBatchReader | None = None
        self._ingestion_activations: dict[int, IngestionLayoutActivation] = {}
        self._closed = False

    def _require_operator_context(self) -> tuple[int, int, int]:
        context = getattr(self, "ctx", None)
        if context is None:
            raise RuntimeError("BatchReaderSPMD requires an active operator context")
        step = getattr(context, "step", None)
        logical_rank = getattr(context, "logical_rank", None)
        logical_world = getattr(context, "logical_world_size", None)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise RuntimeError("BatchReaderSPMD requires a non-negative operator step")
        if not isinstance(logical_rank, int) or isinstance(logical_rank, bool):
            raise TypeError("BatchReaderSPMD requires a logical rank")
        if (
            not isinstance(logical_world, int)
            or isinstance(logical_world, bool)
            or logical_world <= 0
        ):
            raise RuntimeError("BatchReaderSPMD requires a positive logical world size")
        if logical_rank < 0 or logical_rank >= logical_world:
            raise RuntimeError(
                "BatchReaderSPMD logical rank is outside the active world"
            )
        return step, logical_rank, logical_world

    def _to_transport_batch(
        self, batch: Mapping[str, Any], logical_rank: int, logical_world: int
    ) -> Mapping[str, TransportTensor]:
        converted: dict[str, Any] = {}
        for name, value in batch.items():
            if isinstance(value, RaggedFeatureBatch):
                rows = value.row_count
                origins, global_rows = self._source_batch_coordinates(rows)
                converted[name] = wrap_ragged_feature_batch(
                    value,
                    global_rows=global_rows,
                    origin_indices=origins,
                    source_step=self.ctx.step,
                )
                continue
            if isinstance(value, TransportTensor):
                if value.origin_indices is not None:
                    converted[name] = value
                    continue
                if value.global_shape is None:
                    raise ValueError(
                        f"training batch field {name!r} lacks global_shape required "
                        "to derive strict origin provenance"
                    )
                global_rows = int(value.global_shape[0])
                if global_rows < 0 or global_rows % logical_world != 0:
                    raise ValueError(
                        f"training batch field {name!r} global row count "
                        f"{global_rows} is not divisible by logical_world={logical_world}"
                    )
                local_rows = global_rows // logical_world
                origins = ContinuousIndex.chunk(logical_rank * local_rows, local_rows)
                converted[name] = TransportTensor(
                    flat_values=value.flat_values,
                    nested_splits=value.nested_splits,
                    origin_indices=origins,
                    routing_indices=value.routing_indices,
                    global_shape=value.global_shape,
                    receipt=value.receipt,
                    routing_axes=value.routing_axes,
                )
                continue

            array = np.asarray(value)
            if array.dtype == object:
                raise TypeError(f"training batch field {name!r} has object dtype")
            bs = array.shape[0] if array.ndim > 0 else 1
            if array.ndim == 0:
                g_shape = (1,)
                origins = None
            origins, g_rows = self._source_batch_coordinates(bs)
            g_shape = (g_rows, *array.shape[1:])
            converted[name] = TransportTensor.from_tensor(
                array,
                global_shape=g_shape,
                origin_indices=origins,
            )

        return converted

    async def read(self) -> Any:
        if self._closed:
            raise RuntimeError("Reader is closed")
        if self.repeat_first_batch and self._cached_batch is not None:
            return self._output_builder(_clone_transport_mapping(self._cached_batch))

        step, logical_rank, logical_world = self._require_operator_context()

        layout_session = next(
            (
                session
                for epoch, session in self._layout_sessions.items()
                if self._ingestion_activations[epoch].start_position
                <= step
                < self._ingestion_activations[epoch].end_position
            ),
            None,
        )
        if layout_session is not None:
            raw_batch = await layout_session.read()
            if raw_batch is None:
                return ExecutionResult(
                    status=ExecutionStatus.SOURCE_EXHAUSTED,
                    control_requests=(layout_session.acknowledgement,),
                )
            batch = self.transform_raw_batch(raw_batch)
            canonical = self._to_transport_batch(batch, logical_rank, logical_world)
            result = self._output_builder(canonical)
            if self.repeat_first_batch and self._cached_batch is None:
                self._cached_batch = canonical
            return result

        if self._assigned_session is not None:
            work_items = self.ctx.source_work_items
            if not work_items:
                raise RuntimeError("assigned Reader source requires source work items")
            raw_batch = await self._assigned_session.read(
                step,
                work_items=work_items,
            )
        else:
            raise RuntimeError("BatchReaderSPMD has no activated Fluxway source")

        if raw_batch is None:
            return ExecutionResult(status=ExecutionStatus.SOURCE_EXHAUSTED)

        batch = self.transform_raw_batch(raw_batch)
        with profiler.scope("reader.to_transport_batch"):
            canonical = self._to_transport_batch(batch, logical_rank, logical_world)
        result = self._output_builder(canonical)

        if getattr(self, "_execution_audit_path", None) is not None:
            self._execution_audit_records.extend(
                {
                    "rank": self.rank,
                    "epoch": item.key.epoch,
                    "step": item.key.step,
                    "assignment_id": item.key.assignment_id,
                    "source_identity": item.assignment.source_identity,
                    "schema_digest": item.assignment.schema_digest,
                    "global_start": item.global_start,
                    "cardinality": item.cardinality,
                    "logical_start": item.logical_start,
                    "logical_end": item.logical_end,
                    "global_batch_size": item.assignment.global_batch_size,
                }
                for item in work_items
            )

        if self.repeat_first_batch and self._cached_batch is None:
            self._cached_batch = canonical
        return result

    def transform_raw_batch(self, raw_batch: Any) -> Mapping[str, Any]:
        """Transform raw batch from storage into a column mapping."""
        if isinstance(raw_batch, pa.Table):
            return {
                col: raw_batch.column(col).to_numpy() for col in raw_batch.column_names
            }
        if isinstance(raw_batch, Mapping):
            return raw_batch
        raise TypeError(
            f"BatchReaderSPMD batch must be an Arrow table or mapping, got {type(raw_batch).__name__}"
        )

    def _source_options(self) -> dict[str, Any]:
        return dict(self._assigned_source_options)

    async def activate_ingestion_layout(
        self, activation: IngestionLayoutActivation
    ) -> None:
        if self._closed:
            raise RuntimeError("Reader is closed")
        if not isinstance(activation, IngestionLayoutActivation):
            raise TypeError("Reader activation requires IngestionLayoutActivation")
        prior = self._ingestion_activations.get(activation.epoch)
        if prior == activation:
            return
        if prior is not None:
            raise ValueError("Reader received a conflicting ingestion activation")
        if self._ingestion_activations:
            previous = self._ingestion_activations[max(self._ingestion_activations)]
            if (
                previous.terminal
                or activation.epoch != previous.epoch + 1
                or activation.start_position != previous.end_position
            ):
                raise ValueError("Reader ingestion activations must be contiguous")

        if self._source.descriptor is None:
            if self._assigned_session is not None:
                raise RuntimeError(
                    "ingestion layout cannot replace an active Reader source"
                )
            layout_session = IngestionLayoutSession(
                activation,
                self._ingestion_read_session,
                batch_size=self._read_chunk_size,
            )
            self._layout_sessions[activation.epoch] = layout_session
            self._ingestion_activations[activation.epoch] = activation
            return

        if self._assigned_session is None:
            if self._source_descriptor is None:
                raise RuntimeError("assigned reader source must provide a descriptor")
            assigned_session = self._source_descriptor.create_session(
                self._source_options(), pad_batch_factory=self._source.pad_batch_factory
            )
            if not isinstance(assigned_session, AssignedSourceBatchReader):
                raise TypeError(
                    "source_descriptor.create_session must return AssignedSourceBatchReader"
                )
            self._assigned_session = assigned_session
        self._ingestion_activations[activation.epoch] = activation

    def capture_checkpoint_state(self, context: SaveSession) -> dict[str, Any]:
        if self.repeat_first_batch:
            raise RuntimeError(
                "repeated first-batch Reader does not support checkpoint"
            )
        position = context.cut.position
        if position is None or isinstance(position, bool) or position < 0:
            raise ValueError("Reader checkpoint requires a non-negative cut position")
        return {"position": position}

    def restore_checkpoint_state(
        self, state: Mapping[str, Any], context: RestoreSession
    ) -> None:
        if self.repeat_first_batch:
            raise RuntimeError(
                "repeated first-batch Reader does not support checkpoint"
            )
        if not isinstance(state, Mapping):
            raise TypeError("Reader checkpoint state must be a mapping")
        if set(state) != {"position"}:
            raise ValueError("Reader checkpoint state fields do not match protocol")
        position = state["position"]
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("Reader checkpoint state has invalid position")
        if context.cut.position != position:
            raise ValueError(
                "Reader checkpoint position differs from checkpoint context: "
                f"state={position} context={context.cut.position}"
            )
        if self._assigned_session is not None or self._layout_sessions:
            raise RuntimeError(
                "controlled Reader source restores through Fluxway layout state"
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sessions = (
            *self._layout_sessions.values(),
            self._assigned_session,
        )
        sessions = tuple(session for session in sessions if session is not None)
        self._layout_sessions.clear()
        self._assigned_session = None
        results = await asyncio.gather(
            *(session.close() for session in sessions),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def on_shutdown(self) -> None:
        await self.close()
        audit_path = getattr(self, "_execution_audit_path", None)
        if audit_path is None:
            return
        if getattr(self, "_execution_audit_written", False):
            raise RuntimeError("Reader execution audit was finalized twice")
        target = Path(audit_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rank_path = target.parent / f"{target.stem}_rank_{self.rank}{target.suffix}"
        if rank_path.exists():
            raise FileExistsError(rank_path)
        rank_path.write_text(
            json.dumps(
                {
                    "rank": self.rank,
                    "records": getattr(self, "_execution_audit_records", []),
                },
                indent=2,
                sort_keys=True,
            )
        )
        self._execution_audit_written = True


def get_batch_reader_class(
    runtime_env: dict[str, Any] | None,
    resource_profile: Any,
    reader_workers: int,
    batch_fields: tuple[str, ...] = (),
    source_descriptor: WorkItemScheduledDatasetSourceDescriptor | None = None,
) -> type:
    from fluxway.common.bind import bind_spmd

    output_sharding = {}
    for f in batch_fields:
        output_sharding[f] = {"type": "static", "axis": ["worker"]}

    return bind_spmd(
        BatchReaderSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(reader_workers,),
        runtime_env=runtime_env,
        checkpoint_components=(source_checkpoint_registration(),),
        is_source=True,
        interleaved=True,
        interleave_unit_size=1,
        read={
            "output_sharding": output_sharding,
            **({"source": source_descriptor} if source_descriptor is not None else {}),
        },
    )


def source_checkpoint_registration() -> Any:
    return StateCheckpointComponent(
        "source",
        lambda owner, context: owner.capture_checkpoint_state(context),
        lambda owner, state, context: owner.restore_checkpoint_state(state, context),
        restore_topology=RestoreTopologyMode.GLOBAL_RESHARD,
    ).registration()


__all__ = [
    "BatchReaderSPMD",
    "get_batch_reader_class",
    "source_checkpoint_registration",
]
