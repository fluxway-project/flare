from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class NamedTokens:
    """Stacked (batch, n_tokens, dim) tensor with name-based access.

    JAX pytree:
      leaves   = [data]    <- only traced value
      aux_data = names     <- static tuple; determines JIT cache key
    """

    data: jnp.ndarray  # (batch, n_tokens, dim)
    names: tuple[str, ...]  # static; order matches data axis 1

    _name_to_idx: dict[str, int] = dataclasses.field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._name_to_idx = {name: index for index, name in enumerate(self.names)}

    @classmethod
    def from_list(cls, named_tensors: list[tuple[str, jnp.ndarray]]) -> NamedTokens:
        names = tuple(n for n, _ in named_tensors)
        data = jnp.stack([t for _, t in named_tensors], axis=1)
        return cls(data=data, names=names)

    @classmethod
    def from_stacked(cls, data: jnp.ndarray, names: Sequence[str]) -> NamedTokens:
        if data.shape[1] != len(names):
            raise ValueError(
                f"data.shape[1]={data.shape[1]} != len(names)={len(names)}"
            )
        return cls(data=data, names=tuple(names))

    def __getitem__(self, name: str) -> jnp.ndarray:
        return self.data[:, self._idx(name), :]

    def select(self, names: Sequence[str]) -> jnp.ndarray:
        return jnp.stack([self.data[:, self._idx(n), :] for n in names], axis=1)

    def concat(self, other: NamedTokens) -> NamedTokens:
        return NamedTokens(
            data=jnp.concatenate([self.data, other.data], axis=1),
            names=self.names + other.names,
        )

    def _idx(self, name: str) -> int:
        try:
            return self._name_to_idx[name]
        except KeyError:
            raise KeyError(f"{name!r} not in NamedTokens. Available: {self.names}")

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"NamedTokens(shape={self.data.shape}, n_tokens={len(self.names)})"

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx


jax.tree_util.register_pytree_node(
    NamedTokens,
    flatten_func=lambda nt: ([nt.data], nt.names),
    unflatten_func=lambda names, leaves: NamedTokens(data=leaves[0], names=names),
)

__all__ = ["NamedTokens"]
