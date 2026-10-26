from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from flare.runtime.diagnostics.execution_trace import (
    ExecutionTraceOptions,
    JsonlExecutionTracer,
    _descriptor_summary,
    _flat_int_values,
    _leading_row_count,
    _metadata_summary,
    _numeric_summary,
    _to_numpy,
    _values,
)

_LOGGER = logging.getLogger(__name__)


def _trace_summary(value: Any, *, include_values: bool) -> dict[str, Any]:
    if value is None:
        return _metadata_summary(None)
    return (
        _numeric_summary(value)
        if include_values
        else _metadata_summary(getattr(value, "flat_values", value))
    )


def _origin_values_for_trace(
    tensor: Any, rows: Any, stage: str, group: str
) -> list[int]:
    origin = getattr(tensor, "origin_indices_materialized", None)
    row_count = _leading_row_count(rows)
    if origin is None:
        if row_count == 0:
            return []
        raise ValueError(
            f"[RecsysExecutionTracer.{stage}:{group}] non-empty tensor requires origin_indices"
        )
    values = _flat_int_values(origin)
    if len(values) != row_count:
        raise ValueError(
            f"[RecsysExecutionTracer.{stage}:{group}] origin_indices length {len(values)} "
            f"!= row_count {row_count}"
        )
    return values


def _routing_rank_counts(routing_result: Any, world_size: int) -> list[int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    ranks = _to_numpy(routing_result).reshape(-1)
    if not np.issubdtype(ranks.dtype, np.integer):
        raise TypeError(f"routing result must contain integer ranks, got {ranks.dtype}")
    if ranks.size and (np.any(ranks < 0) or np.any(ranks >= world_size)):
        raise ValueError("routing result contains a rank outside world_size")
    return np.bincount(
        ranks.astype(np.int64, copy=False), minlength=world_size
    ).tolist()


def _row_fingerprints(value: Any) -> list[float]:
    array = _to_numpy(value)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return [float(item) for item in array.astype(np.float64, copy=False).sum(axis=1)]


class RecsysExecutionTracer(JsonlExecutionTracer):
    def report_sparse_nonfinite(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        boundary: str,
        values: Any,
    ) -> None:
        array = _to_numpy(values)
        if not np.issubdtype(array.dtype, np.floating):
            return
        nan_count = int(np.count_nonzero(np.isnan(array)))
        inf_count = int(np.count_nonzero(np.isinf(array)))
        if nan_count == 0 and inf_count == 0:
            return
        payload = {
            "boundary": boundary,
            "values": _numeric_summary(array),
        }
        _LOGGER.warning(
            f"non-finite sparse FP16 values at {boundary}: "
            f"group={group!r} step={step} rank={rank} "
            f"nan_count={nan_count} inf_count={inf_count}",
        )
        self.record_event(
            stage="sparse_precision_warning",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def input_unique(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        local_rows: int,
        original_ids: Any,
        unique_ids: Any,
        inverse: Any,
        unique_global_positions: Any,
        unique_tensor: Any,
        inverse_tensor: Any,
        world_size: int | None = None,
        routing_result: Any | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "rows": local_rows,
            "nnz": _metadata_summary(original_ids)["size"],
            "unique": _trace_summary(unique_ids, include_values=self.include_values),
            "original": _trace_summary(
                original_ids, include_values=self.include_values
            ),
            "inv": _trace_summary(inverse, include_values=self.include_values),
            "unique_tensor": _descriptor_summary(unique_tensor),
            "inv_tensor": _descriptor_summary(inverse_tensor),
        }
        if routing_result is not None:
            if world_size is None:
                raise ValueError(
                    "world_size is required when routing_result is provided"
                )
            payload["unique_routing_rank_counts"] = _routing_rank_counts(
                routing_result, world_size
            )
        if self.include_values:
            nested_splits = getattr(inverse_tensor, "nested_splits", None)
            nested_splits_values = (
                [_flat_int_values(splits) for splits in nested_splits]
                if nested_splits is not None
                else []
            )
            payload.update(
                unique_values=_flat_int_values(unique_ids),
                original_values=_flat_int_values(original_ids),
                inv_values=_flat_int_values(inverse),
                nested_splits_values=nested_splits_values,
                unique_global_positions=_flat_int_values(unique_global_positions),
            )
        self.record_event(
            stage="input_unique",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def sparse_lookup(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        keys: Any,
        embeddings: Any,
        input_tensor: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "keys": _trace_summary(keys, include_values=self.include_values),
            "emb": _trace_summary(embeddings, include_values=self.include_values),
            "input_tensor": _descriptor_summary(input_tensor),
        }
        if self.include_values:
            payload.update(
                keys_values=_flat_int_values(keys),
                origin_values=_origin_values_for_trace(
                    input_tensor,
                    keys,
                    "sparse_lookup",
                    group,
                ),
                emb_row_fingerprints=_row_fingerprints(embeddings),
                embedding_values=_values(embeddings),
            )
        self.record_event(
            stage="sparse_lookup",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def sparse_update(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        keys: Any,
        gradients: Any,
        embeddings_after: Any,
        optimizer_applied: bool,
        gradient_tensor: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "keys": _trace_summary(keys, include_values=self.include_values),
            "grad": _trace_summary(gradients, include_values=self.include_values),
            "emb_after": _trace_summary(
                embeddings_after, include_values=self.include_values
            ),
            "optimizer_applied": optimizer_applied,
            "gradient_tensor": _descriptor_summary(gradient_tensor),
        }
        if self.include_values:
            payload.update(
                keys_values=_flat_int_values(keys),
                gradient_values=_values(gradients),
                embedding_values_after=_values(embeddings_after),
            )
        self.record_event(
            stage="sparse_update",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def sparse_commit(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        keys: Any,
        accumulated_gradients: Any,
        embeddings_after: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "keys": _trace_summary(keys, include_values=self.include_values),
            "accumulated_gradients": _trace_summary(
                accumulated_gradients, include_values=self.include_values
            ),
            "emb_after": _trace_summary(
                embeddings_after, include_values=self.include_values
            ),
        }
        if self.include_values:
            payload.update(
                keys_values=_flat_int_values(keys),
                gradient_values=_values(accumulated_gradients),
                embedding_values_after=_values(embeddings_after),
            )
        self.record_event(
            stage="sparse_commit",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def module_gather(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        embeddings: Any,
        inverse: Any,
        sparse_input: Any,
        embedding_tensor: Any,
        inverse_tensor: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "emb": _trace_summary(embeddings, include_values=self.include_values),
            "inv": _trace_summary(inverse, include_values=self.include_values),
            "sparse_input": _trace_summary(
                sparse_input, include_values=self.include_values
            ),
            "embedding_tensor": _descriptor_summary(embedding_tensor),
            "inverse_tensor": _descriptor_summary(inverse_tensor),
        }
        if self.include_values:
            payload.update(
                embedding_values=_values(embeddings),
                inverse_values=_values(inverse),
                sparse_input_values=_values(sparse_input),
            )
        self.record_event(
            stage="model_gather",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def module_gradient(
        self,
        *,
        group: str,
        step: int | None,
        spmd_id: str,
        rank: int,
        gradient: Any,
        inverse: Any,
        embedding_tensor: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "gradient": _trace_summary(gradient, include_values=self.include_values),
            "inverse": _trace_summary(inverse, include_values=self.include_values),
            "embedding_tensor": _descriptor_summary(embedding_tensor),
        }
        if self.include_values:
            payload.update(
                gradient_values=_values(gradient),
                inverse_values=_values(inverse),
            )
        self.record_event(
            stage="model_gradient",
            group=group,
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )

    def module_update(
        self,
        *,
        step: int | None,
        spmd_id: str,
        rank: int,
        micro_step: int,
        grad_accum_steps: int,
        optimizer_applied: bool,
        module_before: Mapping[str, Any],
        module_after: Mapping[str, Any],
        accumulated_gradients: Mapping[str, Any] | None,
        raw_gradients: Mapping[str, Any] | None,
        reduced_gradients: Mapping[str, Any] | None,
    ) -> None:
        payload: dict[str, Any] = {
            "micro_step": micro_step,
            "grad_accum_steps": grad_accum_steps,
            "optimizer_applied": optimizer_applied,
            "model_before": module_before,
            "model_after": module_after,
            "accumulated_gradients": accumulated_gradients,
            "raw_gradients": raw_gradients,
            "reduced_gradients": reduced_gradients,
        }
        self.record_event(
            stage="model_update",
            group="",
            step=step,
            spmd_id=spmd_id,
            rank=rank,
            payload=payload,
        )


def create_recsys_execution_tracer(
    options: ExecutionTraceOptions,
) -> RecsysExecutionTracer | None:
    if not isinstance(options, ExecutionTraceOptions):
        raise TypeError("options must be ExecutionTraceOptions")
    if not options.enabled:
        return None
    if options.directory is None:
        raise RuntimeError("enabled execution trace options require a directory")
    return RecsysExecutionTracer(
        os.path.abspath(options.directory), options.include_values
    )


__all__ = [
    "ExecutionTraceOptions",
    "RecsysExecutionTracer",
    "create_recsys_execution_tracer",
]
