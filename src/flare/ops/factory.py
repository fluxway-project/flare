from __future__ import annotations

from typing import TYPE_CHECKING

from flare.ops.contracts import (
    SPARSE_EMBEDDING_OPERATOR_BACKENDS,
)

if TYPE_CHECKING:
    from flare.kernels.contracts import SparseEmbeddingOperator


def create_sparse_embedding_operator(
    name: str,
    *,
    value_dtype: str,
    gradient_dtype: str,
    empty_cache: bool = False,
) -> SparseEmbeddingOperator:
    if name == "cupy":
        from flare.kernels.backends.gpu_cupy import CupySparseEmbeddingOperator

        return CupySparseEmbeddingOperator(
            value_dtype=value_dtype,
            gradient_dtype=gradient_dtype,
            empty_cache=empty_cache,
        )
    if name == "cpu_numba":
        from flare.kernels.backends.cpu_numba import CpuNumbaSparseEmbeddingOperator

        return CpuNumbaSparseEmbeddingOperator(
            value_dtype=value_dtype,
            gradient_dtype=gradient_dtype,
            empty_cache=empty_cache,
        )
    if name == "jax":
        from flare.kernels.backends.jax_backend import JaxSparseEmbeddingOperator

        return JaxSparseEmbeddingOperator(
            value_dtype=value_dtype, gradient_dtype=gradient_dtype
        )
    raise ValueError(
        f"Unknown sparse embedding operator backend {name!r}; "
        f"expected one of {SPARSE_EMBEDDING_OPERATOR_BACKENDS}."
    )


__all__ = [
    "create_sparse_embedding_operator",
]
