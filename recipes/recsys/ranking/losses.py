from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
from flax import struct


@struct.dataclass
class MultiTargetCrossEntropyLoss:
    target_labels: np.ndarray
    target_weights: np.ndarray
    class_weights: np.ndarray
    targets: tuple[str, ...] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        sample_weight_map: Mapping[str, Any],
        label_map: Mapping[str, int],
        *,
        target_names: tuple[str, ...],
    ) -> MultiTargetCrossEntropyLoss:
        if not label_map or set(label_map.values()) != set(range(len(label_map))):
            raise ValueError("label indexes must be distinct and contiguous from zero")
        if any(type(index) is not int for index in label_map.values()):
            raise TypeError("label indexes must be integers")
        if not target_names or any(
            not isinstance(name, str) or not name for name in target_names
        ):
            raise ValueError("target_names must contain non-empty target names")
        if len(set(target_names)) != len(target_names) or set(target_names) != set(
            sample_weight_map
        ):
            raise ValueError(
                "target_names must cover the weight configuration exactly once"
            )
        num_labels = len(label_map)
        targets = tuple(target_names)
        target_labels = np.zeros((len(targets), num_labels), dtype=np.float32)
        target_weights = np.zeros((len(targets), num_labels), dtype=np.float32)
        class_weights = np.zeros((len(targets),), dtype=np.float32)

        for target_index, target in enumerate(targets):
            config = sample_weight_map[target]
            class_weight = config.get("class_weight", 1.0)
            if not np.isfinite(class_weight):
                raise ValueError(f"Class weight for '{target}' must be finite.")
            class_weights[target_index] = class_weight
            for label, weight in config["sample_weight"].items():
                if label not in label_map:
                    raise KeyError(f"label '{label}' not found in label_map.")
                if not np.isfinite(weight):
                    raise ValueError(f"Weight for '{label}' must be finite.")
                label_index = label_map[label]
                target_labels[target_index, label_index] = float(weight > 0)
                target_weights[target_index, label_index] = abs(float(weight))

        return cls(
            target_labels=target_labels,
            target_weights=target_weights,
            class_weights=class_weights,
            targets=targets,
        )

    def __call__(
        self,
        logits: jnp.ndarray,
        label_ids: jnp.ndarray,
        row_weights: jnp.ndarray,
        valid_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if row_weights.ndim != 1 or row_weights.shape[0] != logits.shape[0]:
            raise ValueError(
                "row_weights must be a vector aligned with logits: "
                f"weights={row_weights.shape} logits={logits.shape}"
            )
        labels = jnp.asarray(self.target_labels)[:, label_ids].T
        weights = jnp.asarray(self.target_weights)[:, label_ids].T
        weighted_loss = optax.sigmoid_binary_cross_entropy(logits, labels) * (
            weights * jnp.asarray(self.class_weights)
        )
        if valid_mask is not None:
            if valid_mask.ndim != 1 or valid_mask.shape[0] != logits.shape[0]:
                raise ValueError(
                    "valid_mask must be a 1D vector aligned with logits: "
                    f"mask={valid_mask.shape} logits={logits.shape}"
                )
            mask = jnp.asarray(valid_mask, dtype=jnp.float32)
            per_sample_loss = jnp.sum(weighted_loss, axis=-1) * row_weights * mask
            return jnp.sum(per_sample_loss) / jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(jnp.sum(weighted_loss, axis=-1) * row_weights)


__all__ = ["MultiTargetCrossEntropyLoss"]
