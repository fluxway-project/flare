from dataclasses import dataclass

import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numba.typed import List

from flare.data.pipeline.production.builder.constants import (
    ARRAY_FEATURE_MAX_LENGTH,
    NULL_FID_SENTINEL,
)
from flare.data.pipeline.production.builder.cross.base import register_fid
from flare.data.pipeline.production.builder.cross.protocol import CrossColumnContext
from flare.data.pipeline.production.builder.encoding.base import (
    MASK_48,
    pack_sparse_i64_low_bits,
    sparse_i64_list_array_from_offsets,
)

NULL_FID_MASK: int = NULL_FID_SENTINEL & MASK_48


def _cross_input_i64(encoded: pa.Array, dependency: str, width: int) -> pa.Array:
    if width > 1:
        if encoded.type == pa.list_(pa.int64()):
            return encoded
        if encoded.type == pa.list_(pa.int32()):
            return encoded.cast(pa.list_(pa.int64()), safe=True)
        raise TypeError(
            f"fid cross input {dependency!r} must be list<int32> or "
            f"list<int64>, got {encoded.type}"
        )
    if encoded.type == pa.int64():
        return encoded
    if encoded.type == pa.int32():
        return encoded.cast(pa.int64(), safe=True)
    raise TypeError(
        f"fid cross input {dependency!r} must be int32 or int64, got {encoded.type}"
    )


@numba.njit(nogil=True, cache=True)
def _is_null_fid(value: int) -> bool:
    return (value & MASK_48) == NULL_FID_MASK


@numba.njit(nogil=True, cache=True)
def _cross_many_with_missing_sentinel(
    values,
    offsets,
    scalar_validity,
    list_modes,
    masks,
    shifts,
    row_count,
):
    dependency_count = len(values)
    hashes = np.empty(row_count * ARRAY_FEATURE_MAX_LENGTH, dtype=np.int64)
    output_offsets = np.empty(row_count + 1, dtype=np.int32)
    positions = np.empty(dependency_count, dtype=np.int32)
    starts = np.empty(dependency_count, dtype=np.int32)
    ends = np.empty(dependency_count, dtype=np.int32)
    output_offsets[0] = 0
    output_index = 0

    for row_index in range(row_count):
        has_empty_input = False
        for dependency_index in range(dependency_count):
            if list_modes[dependency_index]:
                start = offsets[dependency_index][row_index]
                end = offsets[dependency_index][row_index + 1]
                if start < end:
                    has_valid_value = False
                    for candidate_index in range(start, end):
                        if not _is_null_fid(values[dependency_index][candidate_index]):
                            has_valid_value = True
                            break
                    if not has_valid_value:
                        has_empty_input = True
                        start = end
            elif scalar_validity[dependency_index][row_index]:
                start = row_index
                end = row_index + 1
            else:
                start = 0
                end = 0
            starts[dependency_index] = start
            ends[dependency_index] = end
            positions[dependency_index] = start
            if start == end:
                has_empty_input = True

        row_size = 0
        while not has_empty_input and row_size < ARRAY_FEATURE_MAX_LENGTH:
            packed = np.int64(0)
            for dependency_index in range(dependency_count):
                position = positions[dependency_index]
                packed |= (
                    values[dependency_index][position] & masks[dependency_index]
                ) << shifts[dependency_index]
            hashes[output_index] = packed
            output_index += 1
            row_size += 1

            dependency_index = dependency_count - 1
            while dependency_index >= 0:
                next_position = positions[dependency_index] + 1
                while next_position < ends[dependency_index]:
                    candidate = values[dependency_index][next_position]
                    if not _is_null_fid(candidate):
                        break
                    seen_null_fid = False
                    for earlier_position in range(
                        starts[dependency_index], next_position
                    ):
                        if _is_null_fid(values[dependency_index][earlier_position]):
                            seen_null_fid = True
                            break
                    if not seen_null_fid:
                        break
                    next_position += 1
                if next_position < ends[dependency_index]:
                    positions[dependency_index] = next_position
                    break
                positions[dependency_index] = starts[dependency_index]
                dependency_index -= 1
            if dependency_index < 0:
                break

        output_offsets[row_index + 1] = output_index
    return hashes, output_offsets


@register_fid("fidCross")
@register_fid("fidCross2")
@register_fid("fidCross3")
@register_fid("fidCross4")
@dataclass(frozen=True, slots=True)
class FidCrossExpr:
    name: str
    dep_names: list[str]
    bit_sizes: list[int]
    dep_widths: list[int]
    slot: int

    def __post_init__(self) -> None:
        dependency_count = len(self.dep_names)
        if not 2 <= dependency_count <= 4:
            raise ValueError(
                "legacy FidCrossN does not materialize five-dependency results; "
                f"Flare rejects that unusable configuration, got {dependency_count}"
            )
        if len(self.bit_sizes) != dependency_count:
            raise ValueError("fid cross bit_sizes must match dep_names")
        if len(self.dep_widths) != dependency_count:
            raise ValueError("fid cross dep_widths must match dep_names")
        if any(bit_size <= 0 for bit_size in self.bit_sizes):
            raise ValueError("fid cross bit sizes must be positive")
        if sum(self.bit_sizes) != 48:
            raise ValueError("fid cross bit sizes must sum to 48")
        if any(width <= 0 for width in self.dep_widths):
            raise ValueError("fid cross dependency widths must be positive")

    def acero_expr(self) -> pc.Expression | None:
        return None

    def table_compute(self, context: CrossColumnContext) -> pa.ChunkedArray | None:
        arrs: list[pa.Array] = []
        for dep, width in zip(self.dep_names, self.dep_widths):
            encoded = context.column(dep).combine_chunks()
            arrs.append(_cross_input_i64(encoded, dep, width))

        shifts = np.cumsum(np.asarray(self.bit_sizes[::-1], dtype=np.int64))[::-1]
        shifts = np.concatenate((shifts[1:], np.asarray([0], dtype=np.int64)))
        masks = np.asarray(
            [(1 << bit_size) - 1 for bit_size in self.bit_sizes], dtype=np.int64
        )
        if all(width == 1 for width in self.dep_widths):
            packed = pc.bit_wise_and(  # pyrefly: ignore[missing-attribute]
                arrs[0], pa.scalar(int(masks[0]), type=pa.int64())
            )
            packed = pc.shift_left(  # pyrefly: ignore[missing-attribute]
                packed, pa.scalar(int(shifts[0]), type=pa.int64())
            )
            for encoded, mask, shift in zip(arrs[1:], masks[1:], shifts[1:]):
                component = pc.bit_wise_and(  # pyrefly: ignore[missing-attribute]
                    encoded,
                    pa.scalar(int(mask), type=pa.int64()),
                )
                component = pc.shift_left(  # pyrefly: ignore[missing-attribute]
                    component,
                    pa.scalar(int(shift), type=pa.int64()),
                )
                packed = pc.bit_wise_or(  # pyrefly: ignore[missing-attribute]
                    packed,
                    component,
                )
            return pa.chunked_array([pack_sparse_i64_low_bits(packed, self.slot)])

        values = List()
        offsets = List()
        scalar_validity = List()
        list_modes = np.asarray(
            [width > 1 for width in self.dep_widths], dtype=np.bool_
        )
        for encoded, list_mode in zip(arrs, list_modes):
            if list_mode:
                values.append(encoded.flatten().to_numpy(zero_copy_only=False))
                offsets.append(np.asarray(encoded.offsets, dtype=np.int32))
                scalar_validity.append(np.empty(0, dtype=np.bool_))
            else:
                values.append(
                    pc.fill_null(encoded, pa.scalar(0, type=pa.int64())).to_numpy(
                        zero_copy_only=False
                    )
                )
                offsets.append(np.empty(0, dtype=np.int32))
                scalar_validity.append(
                    pc.is_valid(encoded).to_numpy(zero_copy_only=False)
                )

        hashes, output_offsets = _cross_many_with_missing_sentinel(
            values,
            offsets,
            scalar_validity,
            list_modes,
            masks,
            shifts,
            len(arrs[0]),
        )
        hashes = hashes[: output_offsets[-1]]
        flat_int64 = pa.Array.from_buffers(
            pa.int64(), len(hashes), [None, pa.py_buffer(hashes)]
        )
        packed = pack_sparse_i64_low_bits(flat_int64, self.slot)
        offsets_arrow = pa.Array.from_buffers(
            pa.int32(), len(output_offsets), [None, pa.py_buffer(output_offsets)]
        )
        return pa.chunked_array(
            [sparse_i64_list_array_from_offsets(offsets_arrow, packed, self.slot)]
        )
