from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from fluxway.common.tensor_backend import JaxBackend

from flare.kernels.contracts import SparseEmbeddingOperator
from flare.ops.contracts import SparsePooledWidths
from flare.ops.sparse_gather import (
    gather_jax,
    gather_pooled_jax,
)
from flare.ops.sparse_scatter import (
    scatter_jax,
    scatter_pooled_jax,
)


class JaxSparseEmbeddingOperator(SparseEmbeddingOperator):
    def __init__(self, value_dtype: str, gradient_dtype: str):
        super().__init__(JaxBackend(), value_dtype, gradient_dtype)

    def array(self, value: Any) -> Any:
        return self._backend.asarray(value, copy=None)

    def to_dlpack(self, value: Any) -> Any:
        if hasattr(value, "__dlpack__"):
            return value.__dlpack__()
        import jax

        return jax.dlpack.to_dlpack(value)

    def inverse(self, value: Any) -> Any:
        return self.array(value).astype(self.inverse_host_dtype, copy=False)

    def gather(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        pad_index: int,
    ) -> Any:
        embedding = self.array(emb_flat)
        inverse_array = self.inverse(inverse)
        inverse_array = self._co_locate(inverse_array, embedding)
        return gather_jax(
            embedding,
            inverse_array,
            batch_size,
            emb_dim,
            pad_index,
        )

    def gather_pooled(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        target_widths: SparsePooledWidths,
        pad_index: int,
    ) -> Any:
        embedding = self.array(emb_flat)
        inverse_array = self.inverse(inverse)
        inverse_array = self._co_locate(inverse_array, embedding)
        return gather_pooled_jax(
            embedding,
            inverse_array,
            batch_size,
            emb_dim,
            target_widths,
            pad_index,
        )

    @staticmethod
    def _co_locate(value: Any, reference: Any) -> Any:
        import jax

        if not isinstance(reference, jax.Array):
            raise TypeError("JAX sparse gather reference must be a jax.Array")
        return jax.device_put(value, reference.sharding)

    def scatter(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
    ) -> Any:
        import jax.numpy as jnp

        output = scatter_jax(
            emb_rows,
            emb_dim,
            inverse,
            pad_index,
            self.array(gradient).astype(jnp.float32),
        )
        return output.astype(self.gradient_dtype)

    def scatter_pooled(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
        target_widths: SparsePooledWidths,
    ) -> Any:
        import jax.numpy as jnp

        output = scatter_pooled_jax(
            emb_rows,
            emb_dim,
            inverse,
            pad_index,
            self.array(gradient).astype(jnp.float32),
            target_widths,
        )
        return output.astype(self.gradient_dtype)

    def execution_context(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def prewarm(self) -> None:
        import jax.numpy as jnp

        embedding = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        inverse = jnp.asarray([0, 4, 2, 1], dtype=jnp.int32)
        gathered = self.gather(embedding, inverse, 4, 3, 4)
        scattered = self.scatter(4, 3, inverse, 4, jnp.ones((4, 3), dtype=jnp.float32))
        if tuple(gathered.shape) != (4, 1, 3) or tuple(scattered.shape) != (4, 3):
            raise RuntimeError(
                "JAX sparse embedding kernel backend prewarm produced invalid shapes: "
                f"gather={gathered.shape}, scatter={scattered.shape}."
            )


__all__ = ["JaxSparseEmbeddingOperator"]
