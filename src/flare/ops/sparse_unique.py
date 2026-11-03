"""CPU sparse-key deduplication kernels.

The insertion loop is intentionally serial: assigning an id in first-occurrence
order is a data dependency.  The kernel releases the GIL, and InputSPMD runs
independent sparse groups concurrently up to the worker's CPU reservation.
"""

from __future__ import annotations

import numba
import numpy as np

_HASH_MULTIPLIER = np.uint64(11400714819323198485)


@numba.njit(nogil=True, cache=True)
def _unique_first_occurrence_kernel(
    values: np.ndarray,
    unique_out: np.ndarray,
    inverse_out: np.ndarray,
    first_indices_out: np.ndarray,
) -> int:
    count = values.size
    capacity = 1
    while capacity * 2 < count * 3:
        capacity <<= 1

    keys = np.empty(capacity, dtype=values.dtype)
    ids = np.empty(capacity, dtype=np.int32)
    occupied = np.zeros(capacity, dtype=np.uint8)
    mask = np.uint64(capacity - 1)
    unique_count = 0

    for index in range(count):
        key = values[index]
        slot = np.int64(np.uint64(key) * _HASH_MULTIPLIER & mask)
        while occupied[slot] and keys[slot] != key:
            slot = np.int64(np.uint64(slot + 1) & mask)
        if not occupied[slot]:
            occupied[slot] = 1
            keys[slot] = key
            ids[slot] = unique_count
            unique_out[unique_count] = key
            first_indices_out[unique_count] = index
            inverse_out[index] = unique_count
            unique_count += 1
        else:
            inverse_out[index] = ids[slot]
    return unique_count


def unique_first_occurrence(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique keys, inverse ids, and first indices in input order."""
    values = np.asarray(values)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.dtype.kind not in "iu":
        raise TypeError(f"sparse unique requires integer keys, got {values.dtype}")
    if values.size == 0:
        return (
            values.copy(),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )
    unique = np.empty(values.size, dtype=values.dtype)
    inverse = np.empty(values.size, dtype=np.int32)
    first_indices = np.empty(values.size, dtype=np.int64)
    unique_count = _unique_first_occurrence_kernel(
        values, unique, inverse, first_indices
    )
    return (
        unique[:unique_count],
        inverse,
        first_indices[:unique_count],
    )


__all__ = ["unique_first_occurrence"]
