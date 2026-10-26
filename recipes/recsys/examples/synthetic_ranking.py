from __future__ import annotations

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from recipes.recsys.ranking.losses import MultiTargetCrossEntropyLoss
from recipes.recsys.ranking.task import RankingTask


@dataclass(frozen=True)
class ModelConfig:
    target_names: tuple[str, ...] = ("engagement", "reply")
    input_width: int = 2
    learning_rate: float = 0.1
    steps: int = 40


def run(config: ModelConfig | None = None) -> dict[str, object]:
    if config is None:
        config = ModelConfig()
    task = RankingTask(
        label_column="outcome",
        group_id_column="visitor",
        objective=MultiTargetCrossEntropyLoss.create(
            target_names=config.target_names,
            label_map={"negative": 0, "positive": 1},
            sample_weight_map={
                "reply": {"sample_weight": {"negative": -1.0, "positive": 1.0}},
                "engagement": {"sample_weight": {"negative": -1.0, "positive": 1.0}},
            },
        ),
    )
    features = jnp.asarray([[-1, -1], [-1, 0], [1, 0], [1, 1]], dtype=jnp.float32)
    labels = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    row_weights = jnp.full((len(labels),), 1 / len(labels))
    weights = jnp.zeros((config.input_width, len(config.target_names)))
    optimizer = optax.sgd(config.learning_rate)
    optimizer_state = optimizer.init(weights)

    def loss(parameters):
        return task.objective(features @ parameters, labels, row_weights)

    @jax.jit
    def train_step(parameters, state):
        value, gradients = jax.value_and_grad(loss)(parameters)
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, value

    initial_loss = float(loss(weights))
    for _ in range(config.steps):
        weights, optimizer_state, _ = train_step(weights, optimizer_state)
    scores = jax.nn.sigmoid(features @ weights)
    return {
        "target_names": task.objective.targets,
        "initial_loss": initial_loss,
        "final_loss": float(loss(weights)),
        "scores": scores.tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
