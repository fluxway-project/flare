from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from flare.jax.training.gradients.values import as_float_gradient


def accumulate_gradients(accumulated: Any | None, gradients: Any) -> Any:
    """Add one dense gradient tree to an accumulation window."""
    normalized = jax.tree_util.tree_map(as_float_gradient, gradients)
    if accumulated is None:
        return normalized
    try:
        return jax.tree_util.tree_map(
            lambda total, current: total + current,
            accumulated,
            normalized,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient trees must have identical structures") from exc


def build_donated_gradient_accumulator(
    *, in_shardings: Any | None = None, out_shardings: Any | None = None
) -> Callable[[Any, Any], Any]:
    """Build a JIT kernel that reuses the existing dense accumulator buffer."""

    def add(accumulated: Any, gradients: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda total, current: total + current,
            accumulated,
            gradients,
        )

    return jax.jit(
        add,
        in_shardings=None if in_shardings is None else (in_shardings, in_shardings),
        out_shardings=out_shardings,
        donate_argnums=(0,),
    )


__all__ = ["accumulate_gradients", "build_donated_gradient_accumulator"]
