from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from flare.training.contracts import (
    BatchFieldSpec,
    BatchSchema,
    PortSpec,
    TaskSpec,
    TensorSpec,
)
from recipes.recsys.ranking.schema.sparse_features import parse_sparse_group_name
from recipes.recsys.ranking.task import RankingTask


@dataclass(frozen=True)
class BaseRankingRecipe(ABC):
    name: str
    task_name: str
    batch_size: int = 4096
    pipeline_depth: int = 2

    @abstractmethod
    def model_builder(self) -> Any: ...

    @abstractmethod
    def ranking_task(self) -> RankingTask: ...

    def model(self) -> Any:
        return self.model_builder().create_module()

    def training_batches(
        self, *, batch_size: int, steps: int
    ) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        mb = self.model_builder()
        inputs, _ = mb.create_dummy_inputs(
            batch_size=batch_size, use_sparse_embeddings=True
        )
        loss_module = self.ranking_task().objective
        label_ids = (
            jnp.arange(batch_size, dtype=jnp.int32) % loss_module.target_labels.shape[1]
        )
        row_weights = jnp.ones((batch_size,), dtype=jnp.float32)
        targets = {
            "labels": label_ids,
            "row_weights": row_weights,
            "target_labels": loss_module.target_labels,
        }
        for _ in range(steps):
            yield inputs, targets

    def task_spec(self) -> TaskSpec:
        mb = self.model_builder()
        objective = self.ranking_task().objective
        targets = objective.targets
        sparse_groups = tuple(mb.pack_meta.sparse_id_groups)
        side_info_dim = sum(length for _, length in mb.pack_meta.side_info_seq)

        fields = {
            "side_info": BatchFieldSpec(
                "side_info",
                TensorSpec(dtype="int32", shape=(self.batch_size, side_info_dim)),
            )
        }
        for group in sparse_groups:
            _, dimension = parse_sparse_group_name(group)
            fields[group] = BatchFieldSpec(
                group,
                TensorSpec(
                    dtype="float32",
                    shape=(
                        self.batch_size,
                        len(mb.pack_meta.sparse_id_groups[group]),
                        dimension,
                    ),
                ),
            )

        fields.update(
            labels=BatchFieldSpec(
                "labels",
                TensorSpec(dtype="int32", shape=(self.batch_size,)),
                is_label=True,
            ),
            row_weights=BatchFieldSpec(
                "row_weights",
                TensorSpec(dtype="float32", shape=(self.batch_size,)),
                is_label=True,
            ),
        )
        output_ports = (
            PortSpec(
                "logits",
                TensorSpec(dtype="float32", shape=(self.batch_size, len(targets))),
            ),
        )

        schema = BatchSchema(fields=fields)

        def loss_fn(scores: Any, labels: Any) -> Any:
            return (
                objective(
                    scores["logits"],
                    labels["labels"],
                    labels["row_weights"],
                )
                / scores["logits"].shape[0]
            )

        return TaskSpec(
            name=self.task_name,
            batch_schema=schema,
            output_ports=output_ports,
            loss=loss_fn,
            metrics={},
        )


__all__ = ["BaseRankingRecipe"]
