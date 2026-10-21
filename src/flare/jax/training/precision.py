from __future__ import annotations

from enum import Enum

import jax.numpy as jnp

from flare.jax.roles import Fp32Param
from flare.runtime.sparse_precision import (
    SPARSE_BFLOAT16_DTYPE,
    SPARSE_FLOAT16_DTYPE,
    SPARSE_FLOAT32_DTYPE,
    SPARSE_NATIVE_GRADIENT_DTYPE,
)


class PrecisionPolicy(str, Enum):
    """Numerical precision policy for dense NNX training.

    The policy deliberately has a closed set of values.  A BF16 policy changes
    only the temporary forward/gradient computation view; the durable model and
    optimizer state remain FP32. Dynamic sparse embeddings have an explicit
    storage contract exposed separately from the dense compute dtype.
    """

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"

    @classmethod
    def fp32(cls) -> PrecisionPolicy:
        return cls.FP32

    @classmethod
    def bf16(cls) -> PrecisionPolicy:
        return cls.BF16

    @classmethod
    def fp16(cls) -> PrecisionPolicy:
        return cls.FP16

    @property
    def compute_dtype(self) -> jnp.dtype:
        if self is PrecisionPolicy.FP32:
            return jnp.dtype(jnp.float32)
        if self is PrecisionPolicy.BF16:
            return jnp.dtype(jnp.bfloat16)
        if self is PrecisionPolicy.FP16:
            return jnp.dtype(jnp.float16)
        raise ValueError(f"unsupported precision policy: {self!r}")

    @property
    def parameter_dtype(self) -> jnp.dtype:
        if self in (PrecisionPolicy.FP32, PrecisionPolicy.BF16, PrecisionPolicy.FP16):
            return jnp.dtype(jnp.float32)
        raise ValueError(f"unsupported precision policy: {self!r}")

    @property
    def optimizer_dtype(self) -> jnp.dtype:
        if self in (PrecisionPolicy.FP32, PrecisionPolicy.BF16, PrecisionPolicy.FP16):
            return jnp.dtype(jnp.float32)
        raise ValueError(f"unsupported precision policy: {self!r}")

    @property
    def sparse_storage_dtype_name(self) -> str:
        """Durable value dtype for dynamic sparse embeddings."""
        if self is PrecisionPolicy.FP32:
            return SPARSE_FLOAT32_DTYPE
        if self in (PrecisionPolicy.BF16, PrecisionPolicy.FP16):
            return SPARSE_FLOAT16_DTYPE
        raise ValueError(f"unsupported precision policy: {self!r}")

    @property
    def sparse_optimizer_dtype_name(self) -> str:
        if self in (PrecisionPolicy.FP32, PrecisionPolicy.BF16, PrecisionPolicy.FP16):
            return SPARSE_FLOAT32_DTYPE
        raise ValueError(f"unsupported precision policy: {self!r}")

    @property
    def sparse_gradient_dtype_name(self) -> str:
        """Native sparse gradient accumulation dtype."""
        if self in (PrecisionPolicy.FP32, PrecisionPolicy.BF16, PrecisionPolicy.FP16):
            return SPARSE_NATIVE_GRADIENT_DTYPE
        raise ValueError(f"unsupported precision policy: {self!r}")

    @property
    def sparse_gradient_transport_dtype_name(self) -> str:
        """SPMD gradient transport dtype before native accumulation."""
        if self is PrecisionPolicy.FP32:
            return SPARSE_FLOAT32_DTYPE
        if self in (PrecisionPolicy.BF16, PrecisionPolicy.FP16):
            return SPARSE_BFLOAT16_DTYPE
        raise ValueError(f"unsupported precision policy: {self!r}")


__all__ = ["Fp32Param", "PrecisionPolicy"]
