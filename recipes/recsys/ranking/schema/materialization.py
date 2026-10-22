from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flare.runtime.fluxway.transport import (
    transport_alias,
    transport_global_shape,
    transport_nested_splits,
    transport_payload,
    transport_row_count,
)
from flare.runtime.fluxway.workers.base import require_operator_step
from flare.runtime.sparse_precision import require_finite_sparse_values
from flare.runtime.sparse_precision import require_sparse_tensor_dtype
from recipes.recsys.ranking.schema.input import materialize_padded_inverse


def materialize_recsys_sparse_inputs(
    *,
    kwargs: Mapping[str, Any],
    sparse_groups: tuple[str, ...],
    side_info_groups: tuple[str, ...],
    sparse_group_widths: Mapping[str, tuple[int, int]],
    sparse_op: Any,
    world_size: int,
    execution_tracer: Any | None,
    rank: int,
    spmd_id: str,
    operator_owner: Any,
    validate_tensor_values: bool = False,
) -> dict[str, Any]:
    if not isinstance(validate_tensor_values, bool):
        raise TypeError("validate_tensor_values must be bool")
    validate = validate_tensor_values

    labels_value = kwargs["labels"]
    valid_count_value = kwargs["samples_valid_count"]
    batch_size = transport_row_count(labels_value)
    valid_count_device = sparse_op.array(transport_payload(valid_count_value))

    if validate:
        valid_count = int(valid_count_device[0])
        global_batch_size = batch_size * world_size
        if valid_count <= 0 or valid_count > global_batch_size:
            raise ValueError(
                "samples_valid_count is outside the global batch: "
                f"count={valid_count} batch_size={global_batch_size}"
            )

    result: dict[str, Any] = {}
    for group in sparse_groups:
        embedding = kwargs[f"emb_{group}"]
        inverse = kwargs[f"inv_{group}"]
        embedding_array = sparse_op.array(transport_payload(embedding))
        if validate and embedding_array.ndim != 2:
            raise ValueError(f"emb_{group} must be a rank-2 matrix")
        padded_inverse = materialize_padded_inverse(
            transport_payload(inverse),
            transport_nested_splits(inverse),
            transport_global_shape(inverse),
            pad_index=int(embedding_array.shape[0]),
            batch_size=batch_size,
            target_widths=sparse_group_widths[group],
            group=group,
            output_dtype=sparse_op.inverse_host_dtype,
            audit=validate_tensor_values,
        )
        admission_inverse = sparse_op.inverse(padded_inverse)
        sparse_input = sparse_op.gather_pooled(
            embedding_array,
            admission_inverse,
            batch_size,
            int(embedding_array.shape[-1]),
            sparse_group_widths[group],
            int(embedding_array.shape[0]),
        )
        result[f"sparse_input_{group}"] = sparse_input
        result[f"padded_inverse_{group}"] = padded_inverse
        result[f"emb_ref_{group}"] = transport_alias(embedding)
        if execution_tracer is not None:
            step = require_operator_step(
                operator_owner, f"{spmd_id}.prepare_sparse_materialize"
            )
            execution_tracer.module_gather(
                group=group,
                step=step,
                spmd_id=spmd_id,
                rank=rank,
                embeddings=transport_payload(embedding),
                inverse=padded_inverse,
                sparse_input=sparse_input,
                embedding_tensor=embedding,
                inverse_tensor=inverse,
            )
        result[f"sparse_input_{group}"] = require_sparse_tensor_dtype(
            result[f"sparse_input_{group}"],
            sparse_op.value_dtype,
            boundary=f"model_gather:{group}",
        )
        if validate_tensor_values:
            result[f"sparse_input_{group}"] = require_finite_sparse_values(
                result[f"sparse_input_{group}"],
                sparse_op.value_dtype,
                boundary=f"model_gather:{group}",
            )
    return {
        **result,
        **{
            group: sparse_op.array(transport_payload(kwargs[group]))
            for group in side_info_groups
        },
        "labels": sparse_op.array(transport_payload(labels_value)),
        "samples_valid_count": valid_count_device,
    }


__all__ = ["materialize_recsys_sparse_inputs"]
