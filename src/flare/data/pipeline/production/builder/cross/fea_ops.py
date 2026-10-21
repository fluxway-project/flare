from dataclasses import dataclass, field

import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from flare.data.pipeline.production.builder.constants import NULL_FID_SENTINEL
from flare.data.pipeline.production.builder.cross.base import register_fea
from flare.data.pipeline.production.builder.cross.protocol import CrossComputeContext


@register_fea("feaCrossSub")
@dataclass(frozen=True, slots=True)
class SubCrossExpr:
    name: str
    dep_names: list[str]
    out_dtype: pa.DataType = field(default_factory=pa.float32)

    def acero_expr(self) -> pc.Expression | None:
        dep1, dep2 = self.dep_names
        return pc.subtract(  # pyrefly: ignore[missing-attribute]
            pc.cast(pc.field(dep1), self.out_dtype, safe=False),
            pc.cast(pc.field(dep2), self.out_dtype, safe=False),
        )

    def table_compute(self, context: CrossComputeContext) -> pa.ChunkedArray | None:
        dep1, dep2 = self.dep_names
        arr1 = pc.cast(context.column(dep1), self.out_dtype, safe=False)
        arr2 = pc.cast(context.column(dep2), self.out_dtype, safe=False)
        return pc.subtract(arr1, arr2)  # pyrefly: ignore[missing-attribute]


@register_fea("feaCrossDiv")
@dataclass(frozen=True, slots=True)
class DivCrossExpr:
    name: str
    dep_names: list[str]
    eps: float = 1e-6
    out_dtype: pa.DataType = field(default_factory=pa.float32)

    def acero_expr(self) -> pc.Expression | None:
        dep1, dep2 = self.dep_names
        return pc.divide(  # pyrefly: ignore[missing-attribute]
            pc.cast(pc.field(dep1), pa.float32(), safe=False),
            pc.add(  # pyrefly: ignore[missing-attribute]
                pc.cast(pc.field(dep2), pa.float32(), safe=False),
                pa.scalar(self.eps, type=pa.float32()),
            ),
        ).cast(self.out_dtype, safe=False)

    def table_compute(self, context: CrossComputeContext) -> pa.ChunkedArray | None:
        dep1, dep2 = self.dep_names
        arr1 = pc.cast(context.column(dep1), self.out_dtype, safe=False)
        arr2 = pc.cast(context.column(dep2), self.out_dtype, safe=False)
        arr2_safe = pc.add(  # pyrefly: ignore[missing-attribute]
            arr2, pa.scalar(self.eps, type=self.out_dtype)
        )
        return pc.divide(arr1, arr2_safe)  # pyrefly: ignore[missing-attribute]


@numba.njit(nogil=True, cache=True)
def intersect_lists(v1, v2, off1, off2, out):
    for i in range(len(off1) - 1):
        start1, end1 = off1[i], off1[i + 1]
        start2, end2 = off2[i], off2[i + 1]
        count = 0
        for i1 in range(start1, end1):
            val1 = v1[i1]
            if val1 == NULL_FID_SENTINEL:
                continue
            for i2 in range(start2, end2):
                if val1 == v2[i2] and v2[i2] != NULL_FID_SENTINEL:
                    count += 1
        out[i] = count


@numba.njit(nogil=True, cache=True)
def intersect_scalar_list(v1, v2, off2, out):
    for i in range(len(off2) - 1):
        start2, end2 = off2[i], off2[i + 1]
        count = 0
        val1 = v1[i]
        if val1 == NULL_FID_SENTINEL:
            out[i] = count
            continue
        for i2 in range(start2, end2):
            if val1 == v2[i2] and v2[i2] != NULL_FID_SENTINEL:
                count += 1
        out[i] = count


@numba.njit(nogil=True, cache=True)
def intersect_list_scalar(v1, v2, off1, out):
    for i in range(len(off1) - 1):
        start1, end1 = off1[i], off1[i + 1]
        count = 0
        val2 = v2[i]
        if val2 == NULL_FID_SENTINEL:
            out[i] = count
            continue
        for i1 in range(start1, end1):
            if v1[i1] != NULL_FID_SENTINEL and v1[i1] == val2:
                count += 1
        out[i] = count


def _aligned_chunk_slices(
    left: pa.ChunkedArray, right: pa.ChunkedArray
) -> list[tuple[pa.Array, pa.Array]]:
    if len(left) != len(right):
        raise ValueError(
            f"cross inputs have different lengths: {len(left)} != {len(right)}"
        )
    aligned: list[tuple[pa.Array, pa.Array]] = []
    left_index = right_index = 0
    left_offset = right_offset = 0
    while left_index < len(left.chunks) and right_index < len(right.chunks):
        left_chunk = left.chunks[left_index]
        right_chunk = right.chunks[right_index]
        length = min(len(left_chunk) - left_offset, len(right_chunk) - right_offset)
        if length <= 0:
            raise RuntimeError("cross chunk alignment reached an empty segment")
        aligned.append(
            (
                left_chunk.slice(left_offset, length),
                right_chunk.slice(right_offset, length),
            )
        )
        left_offset += length
        right_offset += length
        if left_offset == len(left_chunk):
            left_index += 1
            left_offset = 0
        if right_offset == len(right_chunk):
            right_index += 1
            right_offset = 0
    if left_index != len(left.chunks) or right_index != len(right.chunks):
        raise RuntimeError("cross chunk alignment did not consume both inputs")
    return aligned


@register_fea("feaHitCount")
@dataclass(frozen=True, slots=True)
class HitCountCrossExpr:
    name: str
    dep_names: list[str]
    list_left: bool
    list_right: bool
    in_dtype1: pa.DataType | None = None
    in_dtype2: pa.DataType | None = None
    out_dtype: pa.DataType = field(default_factory=pa.float32)

    def acero_expr(self) -> pc.Expression | None:
        if not self.list_left and not self.list_right:
            dep1, dep2 = self.dep_names
            in1 = self.in_dtype1 if self.in_dtype1 is not None else pa.float32()
            in2 = self.in_dtype2 if self.in_dtype2 is not None else pa.float32()

            if in1 == in2:
                common_type = in1
            elif pa.types.is_integer(in1) and pa.types.is_integer(in2):
                common_type = pa.int64()
            else:
                common_type = pa.float32()

            expr1 = (
                pc.cast(pc.field(dep1), common_type, safe=False)
                if in1 != common_type
                else pc.field(dep1)
            )
            expr2 = (
                pc.cast(pc.field(dep2), common_type, safe=False)
                if in2 != common_type
                else pc.field(dep2)
            )

            return pc.if_else(
                pc.and_(
                    pc.and_(
                        pc.is_valid(expr1),
                        pc.not_equal(
                            expr1,
                            pa.scalar(NULL_FID_SENTINEL, type=common_type),
                        ),
                    ),
                    pc.and_(
                        pc.is_valid(expr2),
                        pc.not_equal(
                            expr2,
                            pa.scalar(NULL_FID_SENTINEL, type=common_type),
                        ),
                    ),
                ),
                pc.equal(expr1, expr2),
                pa.scalar(False),
            ).cast(self.out_dtype, safe=False)
        return None

    def table_compute(self, context: CrossComputeContext) -> pa.ChunkedArray | None:
        if not self.list_left and not self.list_right:
            dep1, dep2 = self.dep_names
            # Need to cast to common type before equality comparison
            in1 = self.in_dtype1 if self.in_dtype1 is not None else pa.float32()
            in2 = self.in_dtype2 if self.in_dtype2 is not None else pa.float32()
            if in1 == in2:
                common_type = in1
            elif pa.types.is_integer(in1) and pa.types.is_integer(in2):
                common_type = pa.int64()
            else:
                common_type = pa.float32()

            arr1_cast = context.cast(dep1, common_type)
            arr2_cast = context.cast(dep2, common_type)
            matches = pc.and_(
                pc.and_(
                    pc.is_valid(arr1_cast),
                    pc.not_equal(
                        arr1_cast,
                        pa.scalar(NULL_FID_SENTINEL, type=common_type),
                    ),
                ),
                pc.and_(
                    pc.is_valid(arr2_cast),
                    pc.not_equal(
                        arr2_cast,
                        pa.scalar(NULL_FID_SENTINEL, type=common_type),
                    ),
                ),
            )
            return pc.cast(
                pc.if_else(matches, pc.equal(arr1_cast, arr2_cast), pa.scalar(False)),
                self.out_dtype,
                safe=False,
            )

        dep1, dep2 = self.dep_names

        in_type1 = self.in_dtype1 if self.in_dtype1 is not None else pa.float32()
        in_type2 = self.in_dtype2 if self.in_dtype2 is not None else pa.float32()

        list_type1 = pa.list_(in_type1) if self.list_left else in_type1
        list_type2 = pa.list_(in_type2) if self.list_right else in_type2

        arr1 = context.cast(dep1, list_type1)
        arr2 = context.cast(dep2, list_type2)

        is_int1 = pa.types.is_integer(in_type1)
        is_int2 = pa.types.is_integer(in_type2)
        np_type1 = np.int64 if is_int1 else np.float32
        np_type2 = np.int64 if is_int2 else np.float32

        res_chunks = []
        for c1, c2 in _aligned_chunk_slices(arr1, arr2):
            length = len(c1)
            out_buf = pa.allocate_buffer(length * 4)
            out_arr = np.frombuffer(out_buf, dtype=np.int32)
            flat1 = c1.flatten() if self.list_left else c1
            flat1 = (
                pc.fill_null(
                    flat1,
                    pa.scalar(NULL_FID_SENTINEL, type=in_type1),
                )
                .to_numpy(zero_copy_only=False)
                .astype(np_type1)
            )
            flat2 = c2.flatten() if self.list_right else c2
            flat2 = (
                pc.fill_null(
                    flat2,
                    pa.scalar(NULL_FID_SENTINEL, type=in_type2),
                )
                .to_numpy(zero_copy_only=False)
                .astype(np_type2)
            )

            if self.list_left and self.list_right:
                o1 = np.asarray(c1.offsets).astype(np.int32)
                o2 = np.asarray(c2.offsets).astype(np.int32)
                intersect_lists(flat1, flat2, o1 - o1[0], o2 - o2[0], out_arr)
            elif not self.list_left and self.list_right:
                o2 = np.asarray(c2.offsets).astype(np.int32)
                intersect_scalar_list(flat1, flat2, o2 - o2[0], out_arr)
            elif self.list_left and not self.list_right:
                o1 = np.asarray(c1.offsets).astype(np.int32)
                intersect_list_scalar(flat1, flat2, o1 - o1[0], out_arr)

            out_int32 = pa.Array.from_buffers(pa.int32(), length, [None, out_buf])
            res_chunks.append(pc.cast(out_int32, self.out_dtype, safe=False))

        return pa.chunked_array(res_chunks)
