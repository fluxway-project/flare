from __future__ import annotations

from flare.training.components.sparse_table.contracts import SparseOptimizerConfig


def create_sparse_optimizer_config(
    kind: str = "adam",
    learning_rate: float = 0.01,
    **kwargs: object,
) -> SparseOptimizerConfig:
    return SparseOptimizerConfig(
        kind=kind,
        learning_rate=learning_rate,
        **kwargs,
    )


__all__ = ["create_sparse_optimizer_config"]
