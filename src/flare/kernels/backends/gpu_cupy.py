from __future__ import annotations

import ctypes
import ctypes.util
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
from fluxway.common.tensor_backend import HybridBackend

from flare.kernels.contracts import SparseEmbeddingOperator
from flare.ops.contracts import (
    SparsePooledWidths,
    pooled_sparse_widths,
)


class CupySparseEmbeddingOperator(SparseEmbeddingOperator):
    inverse_host_dtype = np.int32

    def __init__(
        self,
        value_dtype: str,
        gradient_dtype: str,
        empty_cache: bool = True,
    ):
        super().__init__(
            HybridBackend(default_device="cuda"), value_dtype, gradient_dtype
        )
        self.empty_cache = empty_cache
        import cupy

        self.cp = cupy
        self.gather_kernel = cupy.RawKernel(
            r"""
            extern "C" __global__
            void dense_sparse_gather_pad_f32(
                const float* __restrict__ emb,
                const int* __restrict__ inv,
                float* __restrict__ out,
                const long long slots,
                const int dim,
                const int pad_index
            ) {
                const long long tid = blockDim.x * blockIdx.x + threadIdx.x;
                const long long total = slots * (long long)dim;
                if (tid >= total) return;
                const long long slot = tid / dim;
                const int d = (int)(tid - slot * (long long)dim);
                const int row = inv[slot];
                out[tid] = (row == pad_index) ? 0.0f : emb[row * (long long)dim + d];
            }
            """,
            "dense_sparse_gather_pad_f32",
        )
        self.gather_pooled_kernel = cupy.RawKernel(
            r"""
            extern "C" __global__
            void dense_sparse_gather_pool_f32(
                const float* __restrict__ emb,
                const int* __restrict__ inv,
                float* __restrict__ out,
                const long long batch_size,
                const int feature_count,
                const int max_elements,
                const int dim,
                const int pad_index
            ) {
                const long long tid = blockDim.x * blockIdx.x + threadIdx.x;
                const long long total = batch_size * (long long)feature_count * dim;
                if (tid >= total) return;
                const int d = (int)(tid % dim);
                const long long pooled_slot = tid / dim;
                const int feature = (int)(pooled_slot % feature_count);
                const long long batch = pooled_slot / feature_count;
                const long long inverse_base =
                    (batch * feature_count + feature) * (long long)max_elements;
                float sum = 0.0f;
                for (int element = 0; element < max_elements; ++element) {
                    const int row = inv[inverse_base + element];
                    if (row != pad_index) {
                        sum += emb[row * (long long)dim + d];
                    }
                }
                out[tid] = sum;
            }
            """,
            "dense_sparse_gather_pool_f32",
        )
        self.gather_fp16_kernel = cupy.RawKernel(
            r"""
            #include <cuda_fp16.h>
            extern "C" __global__
            void dense_sparse_gather_pad_f16(
                const __half* __restrict__ emb,
                const int* __restrict__ inv,
                __half* __restrict__ out,
                const long long slots,
                const int dim,
                const int pad_index
            ) {
                const long long tid = blockDim.x * blockIdx.x + threadIdx.x;
                const long long total = slots * (long long)dim;
                if (tid >= total) return;
                const long long slot = tid / dim;
                const int d = (int)(tid - slot * (long long)dim);
                const int row = inv[slot];
                out[tid] = (row == pad_index)
                    ? __float2half(0.0f)
                    : emb[row * (long long)dim + d];
            }
            """,
            "dense_sparse_gather_pad_f16",
        )
        self.gather_pooled_fp16_kernel = cupy.RawKernel(
            r"""
            #include <cuda_fp16.h>
            extern "C" __global__
            void dense_sparse_gather_pool_f16(
                const __half* __restrict__ emb,
                const int* __restrict__ inv,
                __half* __restrict__ out,
                const long long batch_size,
                const int feature_count,
                const int max_elements,
                const int dim,
                const int pad_index
            ) {
                const long long tid = blockDim.x * blockIdx.x + threadIdx.x;
                const long long total = batch_size * (long long)feature_count * dim;
                if (tid >= total) return;
                const int d = (int)(tid % dim);
                const long long pooled_slot = tid / dim;
                const int feature = (int)(pooled_slot % feature_count);
                const long long batch = pooled_slot / feature_count;
                const long long inverse_base =
                    (batch * feature_count + feature) * (long long)max_elements;
                float sum = 0.0f;
                for (int element = 0; element < max_elements; ++element) {
                    const int row = inv[inverse_base + element];
                    if (row != pad_index) {
                        sum += __half2float(emb[row * (long long)dim + d]);
                    }
                }
                out[tid] = __float2half_rn(sum);
            }
            """,
            "dense_sparse_gather_pool_f16",
        )
        self.scatter_kernel = cupy.RawKernel(
            r"""
            extern "C" __global__
            void dense_sparse_scatter_add_pad_f32(
                float* __restrict__ out,
                const int* __restrict__ inv,
                const float* __restrict__ grad,
                const long long slots,
                const int dim,
                const int pad_index
            ) {
                const long long tid = blockDim.x * blockIdx.x + threadIdx.x;
                const long long total = slots * (long long)dim;
                if (tid >= total) return;
                const long long slot = tid / dim;
                const int d = (int)(tid - slot * (long long)dim);
                const int row = inv[slot];
                if (row != pad_index) {
                    atomicAdd(&out[row * (long long)dim + d], grad[tid]);
                }
            }
            """,
            "dense_sparse_scatter_add_pad_f32",
        )
        self.scatter_pooled_kernel = cupy.RawKernel(
            r"""
            extern "C" __global__
            void dense_sparse_scatter_pool_add_f32(
                float* __restrict__ out,
                const int* __restrict__ inv,
                const float* __restrict__ grad,
                const long long total_slots,
                const int feature_count,
                const int max_elements,
                const int dim,
                const int pad_index
            ) {
                const long long tid = blockDim.x * blockIdx.x + threadIdx.x;
                const long long total = total_slots * (long long)dim;
                if (tid >= total) return;
                const int d = (int)(tid % dim);
                const long long slot = tid / dim;
                const int row = inv[slot];
                if (row != pad_index) {
                    const long long feature = (slot / (long long)max_elements) % feature_count;
                    const long long batch = slot / ((long long)feature_count * max_elements);
                    const long long grad_index =
                        (batch * (long long)feature_count + feature) * dim + d;
                    atomicAdd(&out[row * (long long)dim + d], grad[grad_index]);
                }
            }
            """,
            "dense_sparse_scatter_pool_add_f32",
        )
        self.stream = cupy.cuda.Stream(non_blocking=True)
        self._block_size = 256
        self._cuda_memset_async = self._load_cuda_memset_async()

    def _load_cuda_memset_async(self) -> Any:
        candidates = ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
        for candidate in candidates:
            try:
                lib = ctypes.CDLL(candidate)
                func = getattr(lib, "cudaMemsetAsync", None)
                if func is not None:
                    func.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_size_t,
                        ctypes.c_void_p,
                    ]
                    func.restype = ctypes.c_int
                    return func
            except OSError:
                continue
        path = ctypes.util.find_library("cudart")
        if path:
            lib = ctypes.CDLL(path)
            func = getattr(lib, "cudaMemsetAsync", None)
            if func is not None:
                func.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_size_t,
                    ctypes.c_void_p,
                ]
                func.restype = ctypes.c_int
                return func
        raise RuntimeError(
            "Failed to load cudaMemsetAsync from libcudart. Ensure CUDA runtime libraries are installed."
        )

    def _fast_zero(self, array: Any, size_in_bytes: int) -> None:
        status = self._cuda_memset_async(
            ctypes.c_void_p(int(array.data.ptr)),
            0,
            ctypes.c_size_t(size_in_bytes),
            ctypes.c_void_p(int(self.stream.ptr)),
        )
        if status != 0:
            raise RuntimeError(f"cudaMemsetAsync failed with status code {status}")

    def _array_on_stream(self, value: Any) -> Any:
        if hasattr(value, "__jax_array__") or (
            hasattr(value, "__class__") and "jax" in value.__class__.__module__
        ):
            if hasattr(value, "__cuda_array_interface__"):
                return self.cp.asarray(value)
            else:
                import numpy as np

                value = np.asarray(value)
        return self._backend.as_tensor(value, device="cuda")

    def _inverse_on_stream(self, value: Any) -> Any:
        return self._array_on_stream(value).astype(self.cp.int32, copy=False)

    def array(self, value: Any) -> Any:
        with self.stream:
            return self._array_on_stream(value)

    def inverse(self, value: Any) -> Any:
        with self.stream:
            return self._inverse_on_stream(value)

    def gather(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        pad_index: int,
    ) -> Any:
        with self.stream:
            embedding = self._array_on_stream(emb_flat).reshape(-1, emb_dim)
            inverse = self._inverse_on_stream(inverse).reshape(-1)
            slots = int(inverse.size)
            total_elements = slots * emb_dim
            if embedding.dtype == self.cp.float16:
                output = self.cp.empty((slots, emb_dim), dtype=self.cp.float16)
                kernel = self.gather_fp16_kernel
            elif embedding.dtype == self.cp.float32:
                output = self.cp.empty((slots, emb_dim), dtype=self.cp.float32)
                kernel = self.gather_kernel
            else:
                raise TypeError(
                    f"sparse gather requires float16 or float32 embeddings, got {embedding.dtype}"
                )
            if total_elements == 0:
                return output.reshape(batch_size, 0, emb_dim)
            grid_size = (total_elements + self._block_size - 1) // self._block_size
            kernel(
                (grid_size,),
                (self._block_size,),
                (
                    embedding.data.ptr,
                    inverse.data.ptr,
                    output.data.ptr,
                    slots,
                    emb_dim,
                    pad_index,
                ),
                stream=self.stream,
            )
            return output.reshape(batch_size, -1, emb_dim)

    def gather_pooled(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        target_widths: SparsePooledWidths,
        pad_index: int,
    ) -> Any:
        with self.stream:
            feature_count, max_elements = pooled_sparse_widths(target_widths)
            embedding = self._array_on_stream(emb_flat).reshape(-1, emb_dim)
            inverse = self._inverse_on_stream(inverse).reshape(-1)
            total_elements = batch_size * feature_count * emb_dim
            if embedding.dtype == self.cp.float16:
                output = self.cp.empty(
                    (batch_size, feature_count, emb_dim), dtype=self.cp.float16
                )
                kernel = self.gather_pooled_fp16_kernel
            elif embedding.dtype == self.cp.float32:
                output = self.cp.empty(
                    (batch_size, feature_count, emb_dim), dtype=self.cp.float32
                )
                kernel = self.gather_pooled_kernel
            else:
                raise TypeError(
                    "sparse pooled gather requires float16 or float32 embeddings, "
                    f"got {embedding.dtype}"
                )
            if total_elements == 0:
                return output
            grid_size = (total_elements + self._block_size - 1) // self._block_size
            kernel(
                (grid_size,),
                (self._block_size,),
                (
                    embedding.data.ptr,
                    inverse.data.ptr,
                    output.data.ptr,
                    batch_size,
                    feature_count,
                    max_elements,
                    emb_dim,
                    pad_index,
                ),
                stream=self.stream,
            )
            return output

    def scatter(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
    ) -> Any:
        with self.stream:
            inverse = self._inverse_on_stream(inverse).reshape(-1)
            gradient = (
                self._array_on_stream(gradient)
                .astype(self.cp.float32, copy=False)
                .reshape(-1, emb_dim)
            )
            output = self.cp.empty((emb_rows, emb_dim), dtype=self.cp.float32)
            if output.nbytes:
                self._fast_zero(output, output.nbytes)
            slots = int(inverse.size)
            total_elements = slots * emb_dim
            if total_elements == 0:
                return output.astype(self.cp.dtype(self.gradient_dtype), copy=False)
            grid_size = (total_elements + self._block_size - 1) // self._block_size
            self.scatter_kernel(
                (grid_size,),
                (self._block_size,),
                (
                    output.data.ptr,
                    inverse.data.ptr,
                    gradient.data.ptr,
                    slots,
                    emb_dim,
                    pad_index,
                ),
                stream=self.stream,
            )
            return output.astype(self.cp.dtype(self.gradient_dtype), copy=False)

    def scatter_pooled(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
        target_widths: SparsePooledWidths,
    ) -> Any:
        with self.stream:
            feature_count, max_elements = pooled_sparse_widths(target_widths)
            inverse = self._inverse_on_stream(inverse).reshape(-1)
            gradient = (
                self._array_on_stream(gradient)
                .astype(self.cp.float32, copy=False)
                .reshape(-1, feature_count, emb_dim)
            )
            slots = int(inverse.size)
            output = self.cp.empty((emb_rows, emb_dim), dtype=self.cp.float32)
            if output.nbytes:
                self._fast_zero(output, output.nbytes)
            total_elements = slots * emb_dim
            if total_elements == 0:
                return output.astype(self.cp.dtype(self.gradient_dtype), copy=False)
            grid_size = (total_elements + self._block_size - 1) // self._block_size
            self.scatter_pooled_kernel(
                (grid_size,),
                (self._block_size,),
                (
                    output.data.ptr,
                    inverse.data.ptr,
                    gradient.data.ptr,
                    slots,
                    feature_count,
                    max_elements,
                    emb_dim,
                    pad_index,
                ),
                stream=self.stream,
            )
            return output.astype(self.cp.dtype(self.gradient_dtype), copy=False)

    def execution_context(self) -> AbstractContextManager[Any]:
        import contextlib

        @contextlib.contextmanager
        def _context():
            with self.stream:
                try:
                    yield self.stream
                finally:
                    if self.empty_cache:
                        self.cp.get_default_memory_pool().free_all_blocks()
                        self.cp.get_default_pinned_memory_pool().free_all_blocks()

        return _context()

    def prewarm(self) -> None:
        with self.execution_context():
            embedding = self.cp.arange(12, dtype=self.cp.float32).reshape(4, 3)
            inverse = self.cp.asarray([0, 4, 2, 1], dtype=self.cp.int32)
            gathered = self.gather(embedding, inverse, 4, 3, 4)
            scattered = self.scatter(
                4,
                3,
                inverse,
                4,
                self.cp.ones((4, 3), dtype=self.cp.float32),
            )
        self.stream.synchronize()
        if gathered.shape != (4, 1, 3) or scattered.shape != (4, 3):
            raise RuntimeError(
                "CuPy sparse embedding kernel backend prewarm produced invalid shapes: "
                f"gather={gathered.shape}, scatter={scattered.shape}."
            )


__all__ = ["CupySparseEmbeddingOperator"]
