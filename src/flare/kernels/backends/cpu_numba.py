from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numba
import numpy as np
from fluxway.common.tensor_backend import HybridBackend

from flare.kernels.contracts import SparseEmbeddingOperator
from flare.ops.contracts import SparsePooledWidths
from flare.ops.sparse_gather import (
    gather_numba,
    gather_pooled_fp16_numba,
    gather_pooled_numba,
)
from flare.ops.sparse_scatter import scatter_numba, scatter_pooled_numba


class CpuNumbaSparseEmbeddingOperator(SparseEmbeddingOperator):
    def __init__(
        self,
        value_dtype: str,
        gradient_dtype: str,
        empty_cache: bool = True,
    ):
        super().__init__(
            HybridBackend(default_device="cpu"), value_dtype, gradient_dtype
        )
        self.empty_cache = empty_cache
        self._numba_thread_limit = self._resolve_numba_thread_limit()

    @staticmethod
    def _resolve_numba_thread_limit() -> int:
        allocated_cpus = None
        try:
            import ray

            resources = ray.get_runtime_context().get_assigned_resources()
            allocated_cpus = resources.get("CPU")
            if allocated_cpus is None:
                allocated_cpus = next(
                    (
                        value
                        for name, value in resources.items()
                        if name.startswith("CPU_group_")
                    ),
                    None,
                )
        except (
            AssertionError,
            AttributeError,
            ImportError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            allocated_cpus = None

        current_thread_limit = numba.get_num_threads()
        if allocated_cpus is None:
            return current_thread_limit
        return min(max(1, int(float(allocated_cpus))), current_thread_limit)

    def _ensure_numba_thread_mask(self) -> int:
        if numba.get_num_threads() != self._numba_thread_limit:
            numba.set_num_threads(self._numba_thread_limit)
        return self._numba_thread_limit

    def array(self, value: Any) -> np.ndarray:
        if hasattr(value, "__jax_array__") or (
            hasattr(value, "__class__") and "jax" in value.__class__.__module__
        ):
            import numpy as np

            value = np.asarray(value)
        return self._backend.as_tensor(value, device="cpu")

    def inverse(self, value: Any) -> np.ndarray:
        arr = self.array(value)
        return arr.astype(np.int32, copy=False)

    def gather(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        pad_index: int,
    ) -> np.ndarray:
        self._ensure_numba_thread_mask()
        embedding = self.array(emb_flat).reshape(-1, emb_dim)
        inverse = self.inverse(inverse).reshape(-1)
        return gather_numba(
            embedding.reshape(-1), inverse, batch_size, emb_dim, pad_index
        )

    def gather_pooled(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        target_widths: SparsePooledWidths,
        pad_index: int,
    ) -> np.ndarray:
        self._ensure_numba_thread_mask()
        embedding = self.array(emb_flat).reshape(-1, emb_dim)
        inverse = self.inverse(inverse).reshape(-1)
        if embedding.dtype == np.float16:
            return gather_pooled_fp16_numba(
                embedding,
                inverse,
                batch_size,
                emb_dim,
                target_widths,
                pad_index,
            )
        return gather_pooled_numba(
            embedding.reshape(-1),
            inverse,
            batch_size,
            emb_dim,
            target_widths,
            pad_index,
        )

    def scatter(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
    ) -> np.ndarray:
        self._ensure_numba_thread_mask()
        inverse = self.inverse(inverse).reshape(-1)
        gradient = (
            self.array(gradient).astype(np.float32, copy=False).reshape(-1, emb_dim)
        )
        output = scatter_numba(emb_rows, emb_dim, inverse, pad_index, gradient)
        return output.astype(self.gradient_dtype, copy=False)

    def scatter_pooled(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
        target_widths: SparsePooledWidths,
    ) -> np.ndarray:
        self._ensure_numba_thread_mask()
        inverse = self.inverse(inverse).reshape(-1)
        gradient = self.array(gradient).astype(np.float32, copy=False)
        output = scatter_pooled_numba(
            emb_rows,
            emb_dim,
            inverse,
            pad_index,
            gradient,
            target_widths,
        )
        return output.astype(self.gradient_dtype, copy=False)

    def execution_context(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def prewarm(self) -> None:
        self._ensure_numba_thread_mask()
        embedding = np.arange(12, dtype=np.float32).reshape(4, 3)
        inverse = np.asarray([0, 4, 2, 1], dtype=np.int32)
        gathered = self.gather(embedding, inverse, 4, 3, 4)
        scattered = self.scatter(4, 3, inverse, 4, np.ones((4, 3), dtype=np.float32))
        if gathered.shape != (4, 1, 3) or scattered.shape != (4, 3):
            raise RuntimeError(
                "CPU Numba sparse embedding kernel backend prewarm produced invalid shapes: "
                f"gather={gathered.shape}, scatter={scattered.shape}."
            )


__all__ = ["CpuNumbaSparseEmbeddingOperator"]
