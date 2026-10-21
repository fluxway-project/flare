from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RaggedFeatureBatch:
    """Transport-neutral row-partitioned sparse feature payload."""

    flat_values: np.ndarray
    nested_splits: tuple[np.ndarray, ...]
    row_count: int
    feature_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.flat_values, np.ndarray):
            raise TypeError("flat_values must be a NumPy array")
        if self.flat_values.ndim != 1:
            raise ValueError("flat_values must be one-dimensional")
        if not self.flat_values.flags.writeable:
            raise ValueError("flat_values must be writable")
        if type(self.row_count) is not int or self.row_count < 0:
            raise ValueError("row_count must be a non-negative integer")
        splits = self.nested_splits
        if not isinstance(splits, tuple):
            raise TypeError("nested_splits must be a tuple")
        if not splits:
            raise ValueError("nested_splits must not be empty")
        if any(
            not isinstance(split, np.ndarray)
            or split.ndim != 1
            or split.dtype != np.int32
            or not split.flags.writeable
            for split in splits
        ):
            raise ValueError(
                "nested_splits must be writable one-dimensional int32 arrays"
            )
        if splits[0].size != self.row_count + 1:
            raise ValueError("first nested split does not match row_count")
        if any(
            split.size == 0 or int(split[0]) != 0 or np.any(split[1:] < split[:-1])
            for split in splits
        ):
            raise ValueError("nested_splits must be monotonic and start at zero")
        if int(splits[-1][-1]) != self.flat_values.size:
            raise ValueError("last nested split does not match flat_values")
        shape = self.feature_shape
        if not isinstance(shape, tuple):
            raise TypeError("feature_shape must be a tuple")
        if not shape or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("feature_shape must contain non-negative dimensions")


__all__ = ["RaggedFeatureBatch"]
