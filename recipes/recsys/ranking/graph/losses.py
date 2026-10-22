from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp


class RankingLossAdapter:
    def __init__(self, base_loss_fn: Callable[..., Any]) -> None:
        if not callable(base_loss_fn):
            raise TypeError("base_loss_fn must be callable")
        self.base_loss_fn = base_loss_fn

    def __call__(
        self, predictions: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> Any:
        logits = predictions["logits"]
        labels = targets["labels"]
        valid_count = jnp.asarray(targets["samples_valid_count"])
        row_start = jnp.asarray(targets["global_row_start"], dtype=jnp.int32)
        world_size = jnp.asarray(targets["world_size"], dtype=jnp.int32)
        if (
            len(logits.shape) != 2
            or len(labels.shape) != 1
            or labels.shape[0] != logits.shape[0]
        ):
            raise ValueError("ranking logits and labels must be row-aligned")
        if valid_count.shape != (1,) or row_start.shape != () or world_size.shape != ():
            raise ValueError("ranking validity and topology fields have invalid shape")
        valid = (
            row_start + jnp.arange(logits.shape[0], dtype=jnp.int32) < valid_count[0]
        )
        # This is the sample-mean contract: local weights are W/G, so dense
        # pmean produces sum(sample_gradients) / G. Sparse sum or token-mean
        # semantics are different optimizer contracts and must not be inferred
        # from this weight alone.
        weights = (
            valid.astype(logits.dtype)
            * world_size.astype(logits.dtype)
            / valid_count[0]
        )
        return self.base_loss_fn(logits, labels, weights)


__all__ = ["RankingLossAdapter"]
