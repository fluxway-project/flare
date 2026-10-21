from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import sparse_embedding as se
from sparse_embedding.checkpoint import PartitionSpec

from flare.runtime.fluxway.workers.dynamic_embedding import (
    register_sparse_partition_resource,
    sparse_partition_resource_strategy,
)
from flare.runtime.sparse_precision import (
    require_key_dtype,
    require_sparse_key_dtype,
)
from fluxway import profiler
from fluxway.common.bind import bind_spmd
from fluxway.common.decorators import requires_worker_process
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.operator_policy import AsyncPolicy, StatePolicy
from fluxway.common.tensor_backend import get_backend, prepare_dlpack_input
from fluxway.common.tensor_components import (
    dense_path_strides,
    nested_global_positions_kernel,
    nested_path_position_maps,
)
from fluxway.common.types.tensor import TransportTensor

logger = logging.getLogger(__name__)


def _unique_global_flat_positions(
    nested_splits,
    flat_first_indices,
    widths,
    origin_indices,
    group_name,
):
    if len(nested_splits) != len(widths):
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] nested levels "
            f"{len(nested_splits)} != dense widths {len(widths)}."
        )
    local_rows, path_offsets = nested_path_position_maps(
        nested_splits, flat_first_indices, widths
    )
    if local_rows.size and int(local_rows.max()) >= origin_indices.shape[0]:
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] local row index "
            f"{int(local_rows.max())} exceeds origin length {origin_indices.shape[0]}."
        )
    out = np.empty(flat_first_indices.shape[0], dtype=np.int32)
    nested_global_positions_kernel(
        local_rows.astype(np.int32, copy=False),
        path_offsets,
        origin_indices,
        dense_path_strides(widths).astype(np.int32, copy=False),
        out,
    )
    return out


def _backend_adopt(values: Any) -> Any:
    backend = get_backend()
    if isinstance(values, np.ndarray):
        native = backend.asarray(values, copy=False, device="cpu")
    elif hasattr(values, "__cuda_array_interface__"):
        native = backend.asarray(values, copy=False, device="cuda")
    else:
        dlpack_device = getattr(values, "__dlpack_device__", None)
        if not callable(dlpack_device):
            raise TypeError(
                f"sparse deduplication requires a CPU or CUDA array, got {type(values).__name__}"
            )
        device_type, _device_id = dlpack_device()
        if device_type == 1:
            native = backend.asarray(values, copy=False, device="cpu")
        elif device_type == 2:
            native = backend.asarray(values, copy=False, device="cuda")
        else:
            raise TypeError(
                f"sparse deduplication received unsupported DLPack device {device_type!r}"
            )
    return native


async def _backend_unique(values: Any) -> tuple[Any, Any, Any]:
    native = values
    if isinstance(native, np.ndarray):
        unique_dlm, inverse_dlm, first_dlm = await se.async_unique(
            prepare_dlpack_input(native.reshape(-1)),
            return_inverse=True,
            return_first_indices=True,
        )
        return (
            np.from_dlpack(unique_dlm),
            np.from_dlpack(inverse_dlm),
            np.from_dlpack(first_dlm),
        )
    try:
        import cupy
    except ModuleNotFoundError as exc:
        raise TypeError(
            f"unsupported sparse deduplication array {type(native).__name__}"
        ) from exc
    if isinstance(native, cupy.ndarray):
        unique, first_indices, inverse = cupy.unique(
            native.reshape(-1),
            return_index=True,
            return_inverse=True,
        )
        order = cupy.argsort(first_indices)
        inverse_remap = cupy.empty_like(order, dtype=cupy.int32)
        inverse_remap[order] = cupy.arange(order.size, dtype=cupy.int32)
        return (
            unique[order],
            inverse_remap[inverse],
            first_indices[order],
        )
    raise TypeError(f"unsupported sparse deduplication array {type(native).__name__}")


@requires_worker_process
class InputSPMD:
    rank: int
    world_size: int
    ctx: Any
    rank0_address: str

    def __init__(
        self,
        sparse_groups: tuple[str, ...],
        InputNodeOutput: type,
        key_dtype: str,
        sparse_partition_assignment_spec: PartitionSpec,
        execution_tracer: Any | None = None,
    ):
        self.sparse_groups = tuple(sparse_groups)
        self.InputNodeOutput = InputNodeOutput
        self.key_dtype = require_key_dtype(key_dtype)
        if not isinstance(sparse_partition_assignment_spec, PartitionSpec):
            raise TypeError("sparse_partition_assignment_spec must be PartitionSpec")
        resource_ids = tuple(
            dict.fromkeys(
                f"{group.rsplit('_dim_', 1)[0]}.assignment"
                for group in self.sparse_groups
            )
        )
        for resource_id in resource_ids:
            register_sparse_partition_resource(resource_id, sparse_partition_assignment_spec)
        if execution_tracer is not None:
            self.execution_tracer = execution_tracer
            prepare_trace = getattr(execution_tracer, "prepare", None)
            if callable(prepare_trace):
                prepare_trace()

    async def process(self, **kwargs: Any) -> Any:
        get_backend()

        async def _sparse_unique(
            g: str, tt: TransportTensor
        ) -> tuple[TransportTensor, TransportTensor]:
            unique_input = _backend_adopt(tt.flat_values)

            try:
                unique_input = require_sparse_key_dtype(
                    unique_input,
                    self.key_dtype,
                    boundary=f"InputSPMD._sparse_unique:{g}",
                )
            except TypeError as error:
                raise TypeError(f"[InputSPMD._sparse_unique:{g}] {error}") from error
            except ValueError as error:
                raise ValueError(f"[InputSPMD._sparse_unique:{g}] {error}") from error

            if (
                tt.origin_indices_materialized is None
                or len(tt.origin_indices_materialized) == 0
            ):
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] tt.origin_indices_materialized "
                    f"cannot be None or empty under strict routing rules."
                )

            if not tt.nested_splits:
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] sparse ids must use the row-partitioned "
                    "contract: flat_values is an occurrence stream and nested_splits[0] "
                    "maps batch rows to occurrence ranges."
                )
            if not tt.global_shape or len(tt.global_shape) != len(tt.nested_splits) + 1:
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] row-partitioned sparse ids require "
                    f"global_shape=(global_rows, *nested_widths), got {tt.global_shape} "
                    f"for {len(tt.nested_splits)} nested levels."
                )
            local_rows = len(tt.origin_indices_materialized)
            if len(tt.nested_splits[0]) - 1 != local_rows:
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] origin length {local_rows} "
                    f"!= sparse row count {len(tt.nested_splits[0]) - 1}."
                )

            backend = get_backend()
            nested_splits = tuple(tt.nested_splits)
            nested_splits_host = tuple(
                np.asarray(backend.to_numpy(splits), dtype=np.int32)
                for splits in nested_splits
            )

            if unique_input.size == 0:
                unique_ids = unique_input.reshape(-1)
                inv_1d = unique_input.reshape(-1).astype(np.int32, copy=False)
                flat_first_indices = unique_input.reshape(-1)
            else:
                with profiler.scope(f"input.unique.{g}"):
                    unique_ids, inv_1d, flat_first_indices = await _backend_unique(
                        unique_input
                    )
                inv_1d = inv_1d.astype(np.int32, copy=False).reshape(-1)

            if unique_ids.ndim != 1:
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] unique must be 1D, got "
                    f"shape={unique_ids.shape}."
                )
            if inv_1d.size != unique_input.size:
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] inverse length {inv_1d.size} "
                    f"does not match occurrence count {unique_input.size}"
                )
            if flat_first_indices.shape != unique_ids.shape:
                raise ValueError(
                    f"[InputSPMD._sparse_unique:{g}] first-index shape "
                    f"{flat_first_indices.shape} does not match unique shape {unique_ids.shape}"
                )
            flat_first_indices_host = np.asarray(
                backend.to_numpy(flat_first_indices), dtype=np.int32
            )

            widths = tuple(int(w) for w in tt.global_shape[1:])
            with profiler.scope(f"input.positions.{g}"):
                global_positions = _unique_global_flat_positions(
                    nested_splits_host,
                    flat_first_indices_host,
                    widths,
                    np.asarray(
                        backend.to_numpy(tt.origin_indices_materialized),
                        dtype=np.int32,
                    ),
                    g,
                )
            unique_global_shape = (int(np.prod(tt.global_shape, dtype=np.int64)),)

            with profiler.scope(f"input.tensors.{g}"):
                unique_tt = TransportTensor(
                    flat_values=unique_ids,
                    origin_indices=ExplicitIndex(global_positions),
                    global_shape=unique_global_shape,
                )
                inv_tt = TransportTensor(
                    flat_values=inv_1d,
                    nested_splits=list(nested_splits),
                    global_shape=tt.global_shape,
                )
            observer = getattr(self, "execution_tracer", None)
            if observer is not None:
                observer.input_unique(
                    group=g,
                    step=getattr(getattr(self, "ctx", None), "step", None),
                    spmd_id="Input",
                    rank=self.rank,
                    local_rows=local_rows,
                    original_ids=unique_input.reshape(-1),
                    unique_ids=unique_ids,
                    inverse=inv_1d,
                    unique_global_positions=global_positions,
                    unique_tensor=unique_tt,
                    inverse_tensor=inv_tt,
                    world_size=self.world_size,
                )
            return unique_tt, inv_tt

        unique_results = await asyncio.gather(
            *(_sparse_unique(g, kwargs[g]) for g in self.sparse_groups)
        )
        results = dict(zip(self.sparse_groups, unique_results, strict=True))

        ret_kwargs = {}
        for g in self.sparse_groups:
            ret_kwargs[f"unique_{g}"] = results[g][0]
            ret_kwargs[f"inv_{g}"] = results[g][1]

        # Pass through remaining non-sparse tensor fields
        for k, v in kwargs.items():
            if k not in self.sparse_groups:
                ret_kwargs[k] = v

        return self.InputNodeOutput(**ret_kwargs)


def get_input_spmd_class(
    sparse_groups: tuple[str, ...],
    runtime_env: dict[str, Any] | None,
    resource_profile: Any,
    batch_shardings: Mapping[str, Mapping[str, Any]],
    key_dtype: str,
    sparse_partition_assignment_spec: PartitionSpec,
    pipeline_depth: int = 1,
) -> type:
    if pipeline_depth <= 0:
        raise ValueError(f"Input pipeline_depth must be positive, got {pipeline_depth}")
    if not batch_shardings:
        raise ValueError("Input batch_shardings must not be empty")
    require_key_dtype(key_dtype)
    if not isinstance(sparse_partition_assignment_spec, PartitionSpec):
        raise TypeError("sparse_partition_assignment_spec must be PartitionSpec")
    for field, sharding in batch_shardings.items():
        if not isinstance(field, str) or not field:
            raise ValueError("Input batch_shardings keys must be non-empty strings")
        if not isinstance(sharding, Mapping):
            raise TypeError(
                "Input batch_shardings values must be mappings, "
                f"got {type(sharding).__name__} for {field!r}"
            )
    input_out_shards = {}
    for g in sparse_groups:
        base_name, _ = g.rsplit("_dim_", 1)
        input_out_shards[f"unique_{g}"] = {
            "type": "dynamic",
            "axis": ["dp"],
            "strategy": sparse_partition_resource_strategy(
                f"{base_name}.assignment"
            ),
        }
        input_out_shards[f"inv_{g}"] = {"type": "static", "axis": ["dp"]}

    input_out_shards.update(
        {field: dict(sharding) for field, sharding in batch_shardings.items()}
    )

    input_in_shards = {g: {"type": "static", "axis": ["dp"]} for g in sparse_groups}
    input_in_shards.update(
        {field: dict(sharding) for field, sharding in batch_shardings.items()}
    )

    return bind_spmd(
        InputSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        runtime_env=runtime_env,
        process={
            "async_policy": AsyncPolicy.pipelined(
                depth=pipeline_depth,
                state=StatePolicy.thread_safe(),
            ),
            "input_sharding": input_in_shards,
            "output_sharding": input_out_shards,
        },
    )


__all__ = ["InputSPMD", "get_input_spmd_class"]
