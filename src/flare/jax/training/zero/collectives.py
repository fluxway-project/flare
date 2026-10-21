from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.tree_util as jtu

from flare.jax.training.state.layout import FlatStateLeafLayout
from flare.training.zero.contracts import ZeroStage


def reduce_microstep_gradients(
    gradients: Sequence[Any],
    layouts: Sequence[FlatStateLeafLayout],
    *,
    stage: ZeroStage,
    axis_name: str,
    world_size: int,
) -> tuple[Any, ...]:
    if len(gradients) != len(layouts):
        raise ValueError("gradient and layout counts must match")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if stage in (ZeroStage.REPLICATED, ZeroStage.OPTIMIZER_STATE):
        return tuple(
            layout.to_padded_gradient(jax.lax.pmean(value, axis_name))
            for value, layout in zip(gradients, layouts, strict=True)
        )
    if stage in (ZeroStage.GRADIENT, ZeroStage.PARAMETER):
        return tuple(
            jax.lax.psum_scatter(
                layout.to_padded_gradient(value).reshape(world_size, -1),
                axis_name,
                scatter_dimension=0,
                tiled=False,
            )
            / world_size
            for value, layout in zip(gradients, layouts, strict=True)
        )
    raise ValueError(f"unsupported ZeRO stage: {stage!r}")


def reduce_accumulated_gradients(
    gradients: Any, *, stage: ZeroStage, axis_name: str
) -> Any:
    if stage is ZeroStage.OPTIMIZER_STATE:
        return jtu.tree_map(lambda value: jax.lax.psum(value, axis_name), gradients)
    if stage in (ZeroStage.GRADIENT, ZeroStage.PARAMETER):
        return jtu.tree_map(
            lambda value: jax.lax.psum_scatter(
                value, axis_name, scatter_dimension=0, tiled=True
            ),
            gradients,
        )
    if stage is ZeroStage.REPLICATED:
        raise ValueError("replicated state does not require a collective reduction")
    raise ValueError(f"unsupported ZeRO stage: {stage!r}")


__all__ = ["reduce_accumulated_gradients", "reduce_microstep_gradients"]
