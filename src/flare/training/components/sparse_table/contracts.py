from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flare.runtime.sparse_precision import (
    require_sparse_gradient_dtype,
    require_sparse_gradient_transport_dtype,
    require_sparse_optimizer_dtype,
    require_sparse_value_dtype,
)


@dataclass(frozen=True)
class SparseOptimizerConfig:
    kind: str = "adam"
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    dynamic_lr_policy: str = "constant"
    cosine_min_lr_ratio: float = 0.0
    cosine_max_score: float = 1.0
    custom_lut: tuple[float, ...] = ()


@dataclass(frozen=True)
class SparseTableSpec:
    name: str
    embedding_dimensions: tuple[int, ...]
    shards: int
    value_dtype: str
    gradient_transport_dtype: str
    gradient_dtype: str
    optimizer_dtype: str
    optimizer: SparseOptimizerConfig = SparseOptimizerConfig()
    reserve_capacity: int = 1000000
    max_staleness_steps: int = 0

    def __post_init__(self) -> None:
        require_sparse_value_dtype(self.value_dtype)
        require_sparse_gradient_transport_dtype(self.gradient_transport_dtype)
        require_sparse_gradient_dtype(self.gradient_dtype)
        require_sparse_optimizer_dtype(self.optimizer_dtype)


class SparseTableProtocol(Protocol):
    def lookup(self, **kwargs: Any) -> Any: ...
    def update(self, *, optimizer_update_window: Any, **kwargs: Any) -> None: ...


__all__ = [
    "SparseOptimizerConfig",
    "SparseTableProtocol",
    "SparseTableSpec",
]
