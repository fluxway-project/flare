from __future__ import annotations

from typing import Any

import numba
import numpy as np

from flare.ops.contracts import (
    SparsePooledWidths,
    pooled_sparse_widths,
)


@numba.njit(parallel=True, fastmath=True, nogil=True)
def scatter_numba(
    emb_rows: int,
    emb_dim: int,
    inverse: np.ndarray,
    pad_index: int,
    gradient: np.ndarray,
) -> np.ndarray:
    out = np.zeros((emb_rows, emb_dim), dtype=gradient.dtype)
    slots = inverse.shape[0]
    grad = gradient.reshape((-1, emb_dim))

    for d in numba.prange(emb_dim):
        for slot in range(slots):
            row = inverse[slot]
            if row != pad_index:
                out[row, d] += grad[slot, d]

    return out


@numba.njit(parallel=True, fastmath=True, nogil=True)
def _scatter_pooled_numba_kernel(
    emb_rows: int,
    emb_dim: int,
    inverse: np.ndarray,
    pad_index: int,
    gradient: np.ndarray,
    feature_count: int,
    max_elements: int,
) -> np.ndarray:
    out = np.zeros((emb_rows, emb_dim), dtype=gradient.dtype)
    slots = inverse.shape[0]

    for d in numba.prange(emb_dim):
        for slot in range(slots):
            row = inverse[slot]
            if row != pad_index:
                feature = (slot // max_elements) % feature_count
                batch = slot // (feature_count * max_elements)
                out[row, d] += gradient[batch, feature, d]

    return out


def scatter_pooled_numba(
    emb_rows: int,
    emb_dim: int,
    inverse: np.ndarray,
    pad_index: int,
    gradient: np.ndarray,
    target_widths: SparsePooledWidths,
) -> np.ndarray:
    feature_count, max_elements = pooled_sparse_widths(target_widths)
    gradient = gradient.reshape((-1, feature_count, emb_dim))
    return _scatter_pooled_numba_kernel(
        emb_rows,
        emb_dim,
        inverse,
        pad_index,
        gradient,
        feature_count,
        max_elements,
    )


def scatter_jax(
    emb_rows: int,
    emb_dim: int,
    inverse: Any,
    pad_index: int,
    gradient: Any,
) -> Any:
    import jax.numpy as jnp

    inv = inverse.reshape(-1)
    grad = gradient.reshape(-1, emb_dim)
    mask = (inv != pad_index)[:, None]
    safe_inv = jnp.where(inv == pad_index, 0, inv)
    masked_grad = jnp.where(mask, grad, 0.0)
    out = jnp.zeros((emb_rows, emb_dim), dtype=grad.dtype)
    return out.at[safe_inv].add(masked_grad)


def scatter_pooled_jax(
    emb_rows: int,
    emb_dim: int,
    inverse: Any,
    pad_index: int,
    gradient: Any,
    target_widths: SparsePooledWidths,
) -> Any:
    _, max_elements = pooled_sparse_widths(target_widths)
    expanded = gradient[:, :, None, :].repeat(max_elements, axis=2).reshape(-1, emb_dim)
    return scatter_jax(emb_rows, emb_dim, inverse, pad_index, expanded)


__all__ = [
    "scatter_jax",
    "scatter_numba",
    "scatter_pooled_jax",
    "scatter_pooled_numba",
]
