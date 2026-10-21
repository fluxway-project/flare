from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from flare.jax.layers.initialization import require_rngs
from flare.jax.layers.numerics import fp32_softmax


def _get_activation(name: str | None):
    if name is None or name.lower() == "none":
        return None
    act_map = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "sigmoid": jax.nn.sigmoid,
        "softmax": fp32_softmax,
        "tanh": jax.nn.tanh,
    }
    normalized_name = name.lower()
    if normalized_name not in act_map:
        raise ValueError(f"unsupported activation: {name!r}")
    return act_map[normalized_name]


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        dims: list[int],
        activate: str = "relu",
        last_layer_use_activate: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        self.layers = nnx.List()
        for dim in dims:
            self.layers.append(nnx.Linear(in_features, dim, rngs=rngs))
            in_features = dim
        self.dims = dims
        self.activate = activate
        self.last_layer_use_activate = last_layer_use_activate

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = _get_activation(self.activate)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            is_last = i == len(self.layers) - 1
            if act_fn is not None and (not is_last or self.last_layer_use_activate):
                x = act_fn(x)
        return x


__all__ = ["MLP", "_get_activation"]
