from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from flare.jax.training.gradients.values import as_float_gradient


def mean_accumulated_gradients(accumulated: Any, micro_steps: int) -> Any:
    if micro_steps <= 0:
        raise ValueError(f"micro_steps must be positive, got {micro_steps}")
    return jax.tree_util.tree_map(
        lambda value: value / micro_steps,
        accumulated,
    )


def normalize_input_gradients(
    gradients: Mapping[str, Any], *, world_size: int, micro_steps: int
) -> dict[str, Any]:
    """Apply the current sample-mean sparse gradient contract once.

    ``world_size`` matches dense ``pmean`` for sample-weighted loss inputs; it
    is not a universal sparse learning-rate rule. A sparse-sum or token-mean
    contract must use a different explicit owner path. ``micro_steps`` is the
    actual accumulation-window divisor and must not be applied both here and
    again at the sparse optimizer commit boundary.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if micro_steps <= 0:
        raise ValueError(f"micro_steps must be positive, got {micro_steps}")
    return {
        key: as_float_gradient(value) / (world_size * micro_steps)
        for key, value in gradients.items()
    }


__all__ = ["mean_accumulated_gradients", "normalize_input_gradients"]
