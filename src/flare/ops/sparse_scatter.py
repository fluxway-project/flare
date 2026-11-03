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


@numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _scatter_pooled_chunked_numba_kernel(
    emb_rows: int,
    emb_dim: int,
    inverse: np.ndarray,
    pad_index: int,
    gradient: np.ndarray,
    feature_count: int,
    max_elements: int,
    chunk_count: int,
) -> np.ndarray:
    """Scatter in independent slot chunks using bounded thread-local buffers.

    Slot-major traversal reads pooled gradients contiguously. Each chunk owns
    a private dense accumulator, so duplicate IDs never race. A final row-major
    reduction writes contiguous output and combines the partial sums.
    """
    partial = np.zeros((chunk_count, emb_rows, emb_dim), dtype=gradient.dtype)
    slots = inverse.shape[0]
    for chunk in numba.prange(chunk_count):
        start = slots * chunk // chunk_count
        end = slots * (chunk + 1) // chunk_count
        for slot in range(start, end):
            row = inverse[slot]
            if row != pad_index:
                feature = (slot // max_elements) % feature_count
                batch = slot // (feature_count * max_elements)
                for dim in range(emb_dim):
                    partial[chunk, row, dim] += gradient[batch, feature, dim]

    out = np.zeros((emb_rows, emb_dim), dtype=gradient.dtype)
    for row in numba.prange(emb_rows):
        for dim in range(emb_dim):
            value = 0.0
            for chunk in range(chunk_count):
                value += partial[chunk, row, dim]
            out[row, dim] = value
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
    slots = inverse.shape[0]

    # Dimension-parallel scatter has only ``emb_dim`` independent tasks and
    # accesses the row-major gradient with a dimension-sized stride. When this
    # underfills the CPU, split the contiguous slot range into private partial
    # accumulators, then reduce those accumulators by contiguous output rows.
    if slots >= 4096 and emb_dim < max(32, 2 * numba.get_num_threads()):
        bytes_per_chunk = emb_rows * emb_dim * np.dtype(gradient.dtype).itemsize
        # A small cap keeps initialization and final reduction cache-resident;
        # increasing chunk_count for large output tables costs more than it
        # gains in slot-level parallelism.
        scratch_budget = 8 * 1024 * 1024
        chunk_count = min(
            numba.get_num_threads(),
            max(1, scratch_budget // max(1, bytes_per_chunk)),
        )
        return _scatter_pooled_chunked_numba_kernel(
            emb_rows,
            emb_dim,
            inverse,
            pad_index,
            gradient,
            feature_count,
            max_elements,
            int(chunk_count),
        )

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
