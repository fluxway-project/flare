from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _floating_dtype(value: Any) -> jnp.dtype:
    dtype = getattr(value, "dtype", None)
    if dtype is None or not jnp.issubdtype(jnp.dtype(dtype), jnp.floating):
        raise TypeError("numerical precision helpers require a floating-point input")
    return jnp.dtype(dtype)


def _cast(value: Any, dtype: jnp.dtype) -> Any:
    if isinstance(value, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(
            value.shape, dtype, sharding=getattr(value, "sharding", None)
        )
    return jnp.asarray(value).astype(dtype)


def fp32_softmax(value: Any, *, axis: int = -1) -> Any:
    output_dtype = _floating_dtype(value)
    return _cast(jax.nn.softmax(_cast(value, jnp.float32), axis=axis), output_dtype)


def fp32_mean(value: Any, *, axis: Any = None, keepdims: bool = False) -> Any:
    output_dtype = _floating_dtype(value)
    reduced = jnp.mean(_cast(value, jnp.float32), axis=axis, keepdims=keepdims)
    return _cast(reduced, output_dtype)


def fp32_sum(value: Any, *, axis: Any = None, keepdims: bool = False) -> Any:
    output_dtype = _floating_dtype(value)
    reduced = jnp.sum(_cast(value, jnp.float32), axis=axis, keepdims=keepdims)
    return _cast(reduced, output_dtype)


def fp32_rms_normalize(value: Any, *, axis: int = -1, epsilon: float = 1e-5) -> Any:
    output_dtype = _floating_dtype(value)
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    value_fp32 = _cast(value, jnp.float32)
    variance = jnp.mean(jnp.square(value_fp32), axis=axis, keepdims=True)
    normalized = value_fp32 * jax.lax.rsqrt(variance + jnp.float32(epsilon))
    return _cast(normalized, output_dtype)


@jax.custom_vjp
def _fp32_weight_gradient_dot(lhs: Any, rhs: Any) -> Any:
    return jax.lax.dot_general(lhs, rhs, (((1,), (0,)), ((), ())))


def _fp32_weight_gradient_dot_fwd(lhs: Any, rhs: Any) -> tuple[Any, tuple[Any, Any]]:
    return _fp32_weight_gradient_dot(lhs, rhs), (lhs, rhs)


def _fp32_weight_gradient_dot_bwd(
    residual: tuple[Any, Any], cotangent: Any
) -> tuple[Any, Any]:
    lhs, rhs = residual
    input_gradient = (cotangent @ rhs.T).astype(lhs.dtype)
    lhs_fp32 = lhs.astype(jnp.float32)
    cotangent_fp32 = cotangent.astype(jnp.float32)
    fp32_gradient = jax.lax.fori_loop(
        0,
        lhs.shape[0],
        lambda row, total: (
            total + lhs_fp32[row, :, None] * cotangent_fp32[row, None, :]
        ),
        jnp.zeros(rhs.shape, dtype=jnp.float32),
    )
    weight_gradient = fp32_gradient.astype(rhs.dtype)
    return input_gradient, weight_gradient


_fp32_weight_gradient_dot.defvjp(
    _fp32_weight_gradient_dot_fwd, _fp32_weight_gradient_dot_bwd
)


def fp32_weight_gradient_dot_general(
    lhs: Any,
    rhs: Any,
    dimension_numbers: Any,
    *,
    precision: Any = None,
    out_sharding: Any = None,
    preferred_element_type: Any = None,
) -> Any:
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("fp32 weight-gradient dot requires two rank-2 operands")
    if dimension_numbers != (((1,), (0,)), ((), ())):
        raise ValueError("fp32 weight-gradient dot requires dense linear dimensions")
    if lhs.dtype != rhs.dtype or not jnp.issubdtype(lhs.dtype, jnp.floating):
        raise TypeError("fp32 weight-gradient dot requires matching floating dtypes")
    if (
        precision is not None
        or out_sharding is not None
        or preferred_element_type is not None
    ):
        raise ValueError("fp32 weight-gradient dot does not support custom dot options")
    return _fp32_weight_gradient_dot(lhs, rhs)


__all__ = [
    "fp32_mean",
    "fp32_rms_normalize",
    "fp32_softmax",
    "fp32_sum",
    "fp32_weight_gradient_dot_general",
]
