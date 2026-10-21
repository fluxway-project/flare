from __future__ import annotations

import math
from typing import Any

import numba
import numpy as np

from flare.ops.contracts import (
    SparsePooledWidths,
    pooled_sparse_widths,
)


@numba.njit(parallel=True, fastmath=True, nogil=True)
def gather_numba(
    emb_flat: np.ndarray,
    inverse: np.ndarray,
    batch_size: int,
    emb_dim: int,
    pad_index: int,
) -> np.ndarray:
    slots = inverse.shape[0]
    out = np.empty((slots, emb_dim), dtype=emb_flat.dtype)
    emb = emb_flat.reshape((-1, emb_dim))

    for slot in numba.prange(slots):
        row = inverse[slot]
        if row == pad_index:
            for d in range(emb_dim):
                out[slot, d] = 0.0
        else:
            for d in range(emb_dim):
                out[slot, d] = emb[row, d]

    return out.reshape((batch_size, -1, emb_dim))


@numba.njit(parallel=True, fastmath=True, nogil=True)
def _gather_pooled_numba_kernel(
    emb_flat: np.ndarray,
    inverse: np.ndarray,
    batch_size: int,
    feature_count: int,
    max_elements: int,
    emb_dim: int,
    pad_index: int,
) -> np.ndarray:
    out = np.zeros((batch_size, feature_count, emb_dim), dtype=emb_flat.dtype)
    emb = emb_flat.reshape((-1, emb_dim))
    total_slots = batch_size * feature_count

    for idx in numba.prange(total_slots):
        b = idx // feature_count
        f = idx % feature_count
        base = (b * feature_count + f) * max_elements
        for el in range(max_elements):
            row = inverse[base + el]
            if row != pad_index:
                for d in range(emb_dim):
                    out[b, f, d] += emb[row, d]

    return out


def gather_pooled_numba(
    emb_flat: np.ndarray,
    inverse: np.ndarray,
    batch_size: int,
    emb_dim: int,
    target_widths: SparsePooledWidths,
    pad_index: int,
) -> np.ndarray:
    feature_count, max_elements = pooled_sparse_widths(target_widths)
    return _gather_pooled_numba_kernel(
        emb_flat,
        inverse,
        batch_size,
        feature_count,
        max_elements,
        emb_dim,
        pad_index,
    )


@numba.njit(inline="always")
def _fp16_bits_to_float32(bits: np.uint16) -> np.float32:
    sign = -1.0 if (bits & np.uint16(0x8000)) != 0 else 1.0
    exponent = (bits >> np.uint16(10)) & np.uint16(0x1F)
    mantissa = bits & np.uint16(0x03FF)
    if exponent == 0:
        value = sign * math.ldexp(float(mantissa), -24)
    elif exponent == np.uint16(0x1F):
        value = math.inf if mantissa == 0 else math.nan
    else:
        value = sign * math.ldexp(1.0 + float(mantissa) / 1024.0, int(exponent) - 15)
    return np.float32(value)


@numba.njit(parallel=True, nogil=True)
def _gather_pooled_fp16_numba_kernel(
    embedding_bits: np.ndarray,
    inverse: np.ndarray,
    batch_size: int,
    feature_count: int,
    max_elements: int,
    emb_dim: int,
    pad_index: int,
) -> np.ndarray:
    output = np.zeros((batch_size, feature_count, emb_dim), dtype=np.float32)
    embedding = embedding_bits.reshape((-1, emb_dim))
    for slot in numba.prange(batch_size * feature_count):
        batch = slot // feature_count
        feature = slot % feature_count
        inverse_base = slot * max_elements
        for element in range(max_elements):
            row = inverse[inverse_base + element]
            if row != pad_index:
                for dim in range(emb_dim):
                    output[batch, feature, dim] += _fp16_bits_to_float32(
                        embedding[row, dim]
                    )
    return output


def gather_pooled_fp16_numba(
    emb_flat: np.ndarray,
    inverse: np.ndarray,
    batch_size: int,
    emb_dim: int,
    target_widths: SparsePooledWidths,
    pad_index: int,
) -> np.ndarray:
    if emb_flat.dtype != np.float16:
        raise TypeError(
            f"FP16 pooled gather requires float16 embeddings, got {emb_flat.dtype}"
        )
    feature_count, max_elements = pooled_sparse_widths(target_widths)
    pooled = _gather_pooled_fp16_numba_kernel(
        emb_flat.view(np.uint16),
        np.asarray(inverse, dtype=np.int32),
        batch_size,
        feature_count,
        max_elements,
        emb_dim,
        pad_index,
    )
    return pooled.astype(np.float16)


def gather_jax(
    emb_flat: Any,
    inverse: Any,
    batch_size: int,
    emb_dim: int,
    pad_index: int,
) -> Any:
    import jax.numpy as jnp

    emb = emb_flat.reshape(-1, emb_dim)
    inv = inverse.reshape(-1)
    mask = (inv != pad_index)[:, None]
    safe_inv = jnp.where(inv == pad_index, 0, inv)
    gathered = emb[safe_inv]
    out = jnp.where(mask, gathered, 0.0)
    return out.reshape(batch_size, -1, emb_dim)


def gather_pooled_jax(
    emb_flat: Any,
    inverse: Any,
    batch_size: int,
    emb_dim: int,
    target_widths: SparsePooledWidths,
    pad_index: int,
) -> Any:
    import jax.numpy as jnp

    feature_count, max_elements = pooled_sparse_widths(target_widths)
    gathered = gather_jax(emb_flat, inverse, batch_size, emb_dim, pad_index)
    reshaped = gathered.reshape(batch_size, feature_count, max_elements, emb_dim)
    if emb_flat.dtype == jnp.float16:
        return reshaped.sum(axis=2, dtype=jnp.float32).astype(jnp.float16)
    return reshaped.sum(axis=2)


__all__ = [
    "SparsePooledWidths",
    "gather_jax",
    "gather_numba",
    "gather_pooled_fp16_numba",
    "gather_pooled_jax",
    "gather_pooled_numba",
    "pooled_sparse_widths",
]
