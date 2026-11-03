from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.tree_util as jtu
from jax.typing import DTypeLike

from flare.jax.training.state.layout import FlatStateLeafLayout
from flare.training.zero.contracts import ZeroStage


def _reduce_gradient_sum(
    value: Any,
    *,
    stage: ZeroStage,
    axis_name: str | tuple[str, str],
    transport_dtype: DTypeLike | None,
    zero_size: int | None = None,
) -> Any:
    """Cast only at the collective boundary, then restore the accumulator dtype."""
    dtype = value.dtype
    wire_value = value.astype(dtype if transport_dtype is None else transport_dtype)
    if stage in (ZeroStage.REPLICATED, ZeroStage.OPTIMIZER_STATE):
        reduced = jax.lax.psum(wire_value, axis_name)
    elif stage in (ZeroStage.GRADIENT, ZeroStage.PARAMETER):
        if isinstance(axis_name, tuple):
            if zero_size is None:
                raise ValueError("node-local gradient reduction requires zero_size")
            # One global collective is faster than a full cross-node psum
            # followed by a second, local reduce-scatter on this topology.
            global_sum = jax.lax.psum(wire_value, axis_name)
            reduced = global_sum.reshape(zero_size, -1)[
                jax.lax.axis_index(axis_name[1])
            ]
        else:
            reduced = jax.lax.psum_scatter(
                wire_value, axis_name, scatter_dimension=0, tiled=True
            )
    else:
        raise ValueError(f"unsupported ZeRO stage: {stage!r}")
    return reduced.astype(dtype)


def reduce_microstep_gradients(
    gradients: Sequence[Any],
    layouts: Sequence[FlatStateLeafLayout],
    *,
    stage: ZeroStage,
    axis_name: str | tuple[str, str],
    world_size: int,
    transport_dtype: DTypeLike | None = None,
) -> tuple[Any, ...]:
    """Reduce logical or already padded gradients, normalizing in accumulator dtype."""
    if len(gradients) != len(layouts):
        raise ValueError("gradient and layout counts must match")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return tuple(
        _reduce_gradient_sum(
            layout.to_padded_gradient(value),
            stage=stage,
            axis_name=axis_name,
            transport_dtype=transport_dtype,
            zero_size=layout.mesh_size,
        )
        / world_size
        for value, layout in zip(gradients, layouts, strict=True)
    )


def reduce_accumulated_gradients(
    gradients: Any,
    *,
    stage: ZeroStage,
    axis_name: str | tuple[str, str],
    transport_dtype: DTypeLike | None = None,
    zero_size: int | None = None,
) -> Any:
    if stage is ZeroStage.REPLICATED:
        raise ValueError("replicated state does not require a collective reduction")
    return jtu.tree_map(
        lambda value: _reduce_gradient_sum(
            value,
            stage=stage,
            axis_name=axis_name,
            transport_dtype=transport_dtype,
            zero_size=zero_size,
        ),
        gradients,
    )


__all__ = ["reduce_accumulated_gradients", "reduce_microstep_gradients"]
