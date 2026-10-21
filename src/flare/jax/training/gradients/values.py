from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def as_float_gradient(value: Any) -> Any:
    dtype = getattr(value, "dtype", None)
    if dtype == jax.dtypes.float0:
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError("float0 gradients must expose a shape")
        return jnp.zeros(shape, dtype=jnp.float32)
    return value


__all__ = ["as_float_gradient"]
