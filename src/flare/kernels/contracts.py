from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
from fluxway.common.tensor_backend import TensorBackend

from flare.runtime.sparse_precision import (
    require_sparse_gradient_dtype,
    require_sparse_value_dtype,
)


class SparseEmbeddingOperator(ABC):
    inverse_host_dtype = np.int32

    def __init__(
        self,
        tensor_backend: TensorBackend,
        value_dtype: str,
        gradient_dtype: str,
    ):
        if not isinstance(tensor_backend, TensorBackend):
            raise TypeError(
                "tensor_backend must implement Fluxway TensorBackend, got "
                f"{type(tensor_backend).__name__}."
            )
        self._backend = tensor_backend
        self.value_dtype = require_sparse_value_dtype(value_dtype)
        self.gradient_dtype = require_sparse_gradient_dtype(gradient_dtype)

    def array(self, value: Any) -> Any:
        return self._backend.as_tensor(value)

    def to_dlpack(self, value: Any) -> Any:
        if hasattr(value, "__dlpack__"):
            return value.__dlpack__()
        raise TypeError(
            f"Value of type {type(value).__name__} does not support DLPack protocol."
        )

    def local_array(self, value: Any) -> Any:
        from flare.kernels.conversion import extract_local_array

        return self.array(extract_local_array(value))

    def inverse(self, value: Any) -> Any:
        return self.array(value).astype(self.inverse_host_dtype, copy=False)

    @abstractmethod
    def gather(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        pad_index: int,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def gather_pooled(
        self,
        emb_flat: Any,
        inverse: Any,
        batch_size: int,
        emb_dim: int,
        target_widths: tuple[int, ...],
        pad_index: int,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def scatter(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def scatter_pooled(
        self,
        emb_rows: int,
        emb_dim: int,
        inverse: Any,
        pad_index: int,
        gradient: Any,
        target_widths: tuple[int, ...],
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def execution_context(self) -> AbstractContextManager[Any]:
        raise NotImplementedError

    @abstractmethod
    def prewarm(self) -> None:
        raise NotImplementedError


__all__ = [
    "SparseEmbeddingOperator",
]
