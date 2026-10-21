from __future__ import annotations

from flare.kernels.backends import (
    CpuNumbaSparseEmbeddingOperator,
    CupySparseEmbeddingOperator,
    JaxSparseEmbeddingOperator,
)
from flare.kernels.contracts import SparseEmbeddingOperator
from flare.kernels.conversion import (
    extract_local_array,
    release_jax_leases,
    to_cupy_array,
    to_dlpack,
    to_host_array,
    to_jax_array,
    to_numpy,
)

__all__ = [
    "CpuNumbaSparseEmbeddingOperator",
    "CupySparseEmbeddingOperator",
    "JaxSparseEmbeddingOperator",
    "SparseEmbeddingOperator",
    "extract_local_array",
    "release_jax_leases",
    "to_cupy_array",
    "to_dlpack",
    "to_host_array",
    "to_jax_array",
    "to_numpy",
]
