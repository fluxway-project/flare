from __future__ import annotations

from typing import Any

import numpy as np

from flare.kernels.conversion import to_host_array


def validate_inverse_range(inverse: Any, pad_index: int, group: str) -> None:
    values = to_host_array(inverse).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Sparse inverse for group {group!r} has no slots")
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum < 0 or maximum > pad_index:
        raise IndexError(
            f"Sparse inverse for group {group!r} is outside [0, {pad_index}]: "
            f"minimum={minimum}, maximum={maximum}"
        )


def materialize_padded_inverse(
    flat_values: Any,
    nested_splits: tuple[Any, ...] | list[Any],
    global_shape: tuple[int, ...] | None,
    *,
    pad_index: int,
    batch_size: int,
    target_widths: tuple[int, ...],
    group: str,
    output_dtype: np.dtype | type[np.integer] = np.int32,
    audit: bool = False,
) -> np.ndarray:
    dtype = np.dtype(output_dtype)
    flat_values = to_host_array(flat_values)
    if audit:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not target_widths or any(width <= 0 for width in target_widths):
            raise ValueError(f"target_widths must be positive, got {target_widths}")
        if dtype.kind not in "iu":
            raise TypeError(f"inverse dtype must be integral, got {dtype}")
        if pad_index < 0 or pad_index > int(np.iinfo(dtype).max):
            raise ValueError(f"pad_index {pad_index} does not fit {dtype}")

    if not nested_splits:
        padded = to_host_array(flat_values)
        actual_widths = tuple(int(width) for width in padded.shape[1:])
        if audit:
            if padded.shape[0] != batch_size:
                raise ValueError(
                    f"Sparse inverse for group {group!r} has {padded.shape[0]} rows, "
                    f"expected {batch_size}"
                )
            if len(actual_widths) != len(target_widths):
                raise ValueError(
                    f"Sparse inverse rank mismatch for group {group!r}: "
                    f"actual={actual_widths}, target={target_widths}"
                )
            if any(
                actual > target
                for actual, target in zip(actual_widths, target_widths, strict=True)
            ):
                raise ValueError(
                    f"Sparse inverse exceeds target widths for group {group!r}: "
                    f"actual={actual_widths}, target={target_widths}"
                )
            validate_inverse_range(padded, pad_index, group)
        output = np.full((batch_size, *target_widths), pad_index, dtype=dtype)
        slices = (slice(None),) + tuple(slice(0, width) for width in actual_widths)
        output[slices] = padded.astype(dtype, copy=False)
        return output

    if audit:
        if not global_shape or len(global_shape) != len(nested_splits) + 1:
            raise ValueError(
                f"Ragged inverse for group {group!r} requires global_shape"
            )
        metadata_widths = tuple(int(width) for width in global_shape[1:])
        if len(metadata_widths) != len(target_widths):
            raise ValueError(
                f"Ragged inverse metadata rank mismatch for group {group!r}: "
                f"metadata={metadata_widths}, target={target_widths}"
            )
        if len(nested_splits[0]) - 1 != batch_size:
            raise ValueError(
                f"Ragged inverse row count differs from batch for group {group!r}"
            )
    splits = tuple(to_host_array(split, dtype=np.int64) for split in nested_splits)
    flat = to_host_array(flat_values)
    if audit:
        validate_inverse_range(flat, pad_index, group)
    if len(splits) != len(target_widths) or len(splits[0]) != batch_size + 1:
        raise ValueError(f"Ragged inverse shape mismatch for group {group!r}")
    # Validate offsets before indexing; unchecked native writes can corrupt a
    # worker instead of reporting invalid transport metadata.
    output = np.full((batch_size, *target_widths), pad_index, dtype=dtype)
    positions = np.arange(batch_size, dtype=np.int64)
    valid = np.ones(batch_size, dtype=bool)
    for level, (split, width) in enumerate(zip(splits, target_widths, strict=True)):
        expected_children = (
            len(splits[level + 1]) - 1 if level + 1 < len(splits) else flat.size
        )
        if (
            split.ndim != 1
            or split.size != positions.size + 1
            or split[0] != 0
            or split[-1] != expected_children
            or np.any(split[1:] < split[:-1])
        ):
            raise ValueError(
                f"Invalid ragged inverse offsets for group {group!r} at level {level}"
            )
        lengths = np.diff(split)
        parents = np.repeat(np.arange(positions.size), lengths)
        local = np.arange(expected_children) - split[parents]
        valid = valid[parents] & (local < width)
        positions = positions[parents] * width + local
    output.reshape(-1)[positions[valid]] = flat.reshape(-1)[valid].astype(
        dtype, copy=False
    )
    return output


__all__ = ["materialize_padded_inverse", "validate_inverse_range"]
