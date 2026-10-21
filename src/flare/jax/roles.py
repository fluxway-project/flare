from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx


class Fp32Param(nnx.Param):
    """Trainable parameter whose compute view remains FP32."""


class ConstantBuffer(nnx.Variable):
    """Immutable non-trainable tensor carried as module state."""


def cast_floating_tree(value: Any, dtype: jnp.dtype) -> Any:
    """Cast floating leaves without runtime inspection or synchronization."""
    target_dtype = jnp.dtype(dtype)

    def cast_leaf(leaf: Any) -> Any:
        source_dtype = getattr(leaf, "dtype", None)
        if source_dtype is None or not jnp.issubdtype(
            jnp.dtype(source_dtype), jnp.floating
        ):
            return leaf
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(
                leaf.shape, target_dtype, sharding=getattr(leaf, "sharding", None)
            )
        return jnp.asarray(leaf).astype(target_dtype)

    return jtu.tree_map(cast_leaf, value)


def cast_floating_tree_checked(
    value: Any,
    dtype: jnp.dtype,
    *,
    boundary: str,
) -> Any:
    """Cast floating leaves after an explicit finite/range diagnostic gate."""
    target_dtype = jnp.dtype(dtype)
    target_limit = (
        jnp.finfo(target_dtype).max if target_dtype == jnp.dtype(jnp.float16) else None
    )

    def cast_leaf(leaf: Any) -> Any:
        source_dtype = getattr(leaf, "dtype", None)
        if source_dtype is None or not jnp.issubdtype(
            jnp.dtype(source_dtype), jnp.floating
        ):
            return leaf
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(
                leaf.shape, target_dtype, sharding=getattr(leaf, "sharding", None)
            )
        array = jnp.asarray(leaf)
        if target_limit is not None and jnp.dtype(array.dtype) != target_dtype:
            invalid = jnp.any(~jnp.isfinite(array) | (jnp.abs(array) > target_limit))

            def raise_if_invalid(flag: Any) -> None:
                if bool(flag):
                    raise FloatingPointError(
                        f"{boundary} received a non-finite or out-of-range FP16 value"
                    )

            if isinstance(invalid, jax.core.Tracer):
                jax.debug.callback(raise_if_invalid, invalid, ordered=True)
            elif bool(invalid):
                raise FloatingPointError(
                    f"{boundary} received a non-finite or out-of-range FP16 value"
                )
        return array.astype(target_dtype)

    return jtu.tree_map(cast_leaf, value)


__all__ = [
    "ConstantBuffer",
    "Fp32Param",
    "cast_floating_tree",
    "cast_floating_tree_checked",
]
