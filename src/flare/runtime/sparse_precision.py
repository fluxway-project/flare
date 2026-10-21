from __future__ import annotations

from typing import Any

import numpy as np
from sparse_embedding.checkpoint import SUPPORTED_KEY_DTYPES

SPARSE_FLOAT16_DTYPE = "float16"
SPARSE_BFLOAT16_DTYPE = "bfloat16"
SPARSE_FLOAT32_DTYPE = "float32"
SPARSE_VALUE_DTYPES = frozenset({SPARSE_FLOAT16_DTYPE, SPARSE_FLOAT32_DTYPE})
SPARSE_GRADIENT_TRANSPORT_DTYPES = frozenset(
    {SPARSE_BFLOAT16_DTYPE, SPARSE_FLOAT32_DTYPE}
)
SPARSE_GRADIENT_DTYPES = frozenset({SPARSE_FLOAT32_DTYPE})
SPARSE_OPTIMIZER_DTYPES = frozenset({SPARSE_FLOAT32_DTYPE})
SPARSE_NATIVE_GRADIENT_DTYPE = SPARSE_FLOAT32_DTYPE


def _require_dtype(value: Any, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)!r}, got {value!r}")
    return value


def require_sparse_value_dtype(value: Any) -> str:
    return _require_dtype(value, SPARSE_VALUE_DTYPES, "sparse embedding value_dtype")


def require_sparse_gradient_dtype(value: Any) -> str:
    return _require_dtype(
        value,
        SPARSE_GRADIENT_DTYPES,
        "sparse embedding gradient_dtype",
    )


def require_sparse_gradient_transport_dtype(value: Any) -> str:
    return _require_dtype(
        value,
        SPARSE_GRADIENT_TRANSPORT_DTYPES,
        "sparse embedding gradient_transport_dtype",
    )


def require_sparse_optimizer_dtype(value: Any) -> str:
    return _require_dtype(
        value, SPARSE_OPTIMIZER_DTYPES, "sparse embedding optimizer_dtype"
    )


def require_key_dtype(value: Any) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_KEY_DTYPES:
        raise ValueError(
            "sparse embedding key_dtype must be one of "
            f"{tuple(sorted(SUPPORTED_KEY_DTYPES))!r}, got {value!r}"
        )
    return value


def require_sparse_key_dtype(values: Any, key_dtype: str, *, boundary: str) -> Any:
    expected_dtype = np.dtype(require_key_dtype(key_dtype))
    allowed_dtypes = SUPPORTED_KEY_DTYPES
    try:
        actual_dtype = np.dtype(values.dtype)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            f"sparse key dtype validation at {boundary} requires an array dtype"
        ) from error
    if actual_dtype.name not in allowed_dtypes:
        raise TypeError(
            f"sparse key dtype at {boundary} must be one of "
            f"{sorted(allowed_dtypes)!r}, got {actual_dtype.name!r}"
        )
    if actual_dtype != expected_dtype:
        raise TypeError(
            f"sparse key dtype mismatch at {boundary}: "
            f"expected={expected_dtype.name}, got={actual_dtype.name}"
        )
    return values


def _numpy_dtype(values: Any) -> np.dtype[Any]:
    module_name = type(values).__module__
    if module_name.startswith("torch"):
        import torch

        torch_dtypes = {
            torch.float16: np.dtype("float16"),
            torch.bfloat16: np.dtype("bfloat16"),
            torch.float32: np.dtype("float32"),
        }
        try:
            return torch_dtypes[values.dtype]
        except KeyError as error:
            raise TypeError(
                f"unsupported sparse tensor dtype: {values.dtype}"
            ) from error
    try:
        return np.dtype(values.dtype)
    except TypeError as error:
        raise TypeError(f"unsupported sparse tensor dtype: {values.dtype}") from error


def require_sparse_tensor_dtype(values: Any, dtype: str, *, boundary: str) -> Any:
    if dtype not in SPARSE_VALUE_DTYPES | SPARSE_GRADIENT_TRANSPORT_DTYPES:
        raise ValueError(
            "sparse precision check requires float16, bfloat16, or float32, "
            f"got {dtype!r}"
        )
    expected_dtype = np.dtype(dtype)
    actual_dtype = _numpy_dtype(values)
    if actual_dtype != expected_dtype:
        raise TypeError(
            f"sparse precision check dtype mismatch at {boundary}: expected={expected_dtype} "
            f"actual={actual_dtype}"
        )
    return values


def require_finite_sparse_values(values: Any, dtype: str, *, boundary: str) -> Any:
    """Inspect values explicitly outside execution workers."""
    require_sparse_tensor_dtype(values, dtype, boundary=boundary)
    module_name = type(values).__module__
    if isinstance(values, np.ndarray):
        finite = bool(np.isfinite(values).all())
    elif module_name.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        finite = bool(jnp.isfinite(values).all())
    elif hasattr(values, "__cuda_array_interface__"):
        import cupy as cp

        if not isinstance(values, cp.ndarray):
            raise TypeError("sparse precision check requires a CuPy CUDA array")
        finite = bool(cp.isfinite(values).all())
    elif module_name.startswith("torch"):
        import torch

        finite = bool(torch.isfinite(values).all())
    else:
        raise TypeError(
            f"unsupported sparse precision check tensor type: {type(values).__name__}"
        )
    if not finite:
        raise FloatingPointError(f"non-finite sparse values at {boundary}")
    return values


def prepare_sparse_native_gradient(
    values: Any, *, transport_dtype: str, native_dtype: str,
    validate_values: bool = False,
) -> Any:
    require_sparse_gradient_transport_dtype(transport_dtype)
    require_sparse_gradient_dtype(native_dtype)
    expected_transport = np.dtype(transport_dtype)
    actual_transport = _numpy_dtype(values)
    if actual_transport != expected_transport:
        raise TypeError(
            f"sparse gradient transport dtype mismatch: expected={expected_transport} "
            f"actual={actual_transport}"
        )
    target_dtype = np.dtype(native_dtype)
    module_name = type(values).__module__
    if isinstance(values, np.ndarray):
        cast_values = values.astype(target_dtype, copy=False)
    elif module_name.startswith(("jax", "jaxlib")):
        cast_values = values.astype(target_dtype)
    elif hasattr(values, "__cuda_array_interface__"):
        cast_values = values.astype(target_dtype, copy=False)
    elif module_name.startswith("torch"):
        import torch

        torch_dtype = {
            np.dtype("float16"): torch.float16,
            np.dtype("float32"): torch.float32,
        }[target_dtype]
        cast_values = values.to(dtype=torch_dtype)
    else:
        raise TypeError(
            f"unsupported sparse gradient tensor type: {type(values).__name__}"
        )
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be bool")
    if validate_values:
        return require_finite_sparse_values(
            cast_values, native_dtype, boundary="sparse_native_gradient"
        )
    return cast_values


__all__ = [
    "SPARSE_BFLOAT16_DTYPE",
    "SPARSE_FLOAT16_DTYPE",
    "SPARSE_FLOAT32_DTYPE",
    "SPARSE_GRADIENT_DTYPES",
    "SPARSE_GRADIENT_TRANSPORT_DTYPES",
    "SPARSE_NATIVE_GRADIENT_DTYPE",
    "SPARSE_OPTIMIZER_DTYPES",
    "SPARSE_VALUE_DTYPES",
    "prepare_sparse_native_gradient",
    "require_finite_sparse_values",
    "require_key_dtype",
    "require_sparse_gradient_dtype",
    "require_sparse_gradient_transport_dtype",
    "require_sparse_key_dtype",
    "require_sparse_optimizer_dtype",
    "require_sparse_tensor_dtype",
    "require_sparse_value_dtype",
]
