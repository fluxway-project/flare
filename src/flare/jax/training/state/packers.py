from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
from flax import nnx


class NnxStatePacker:
    """Flatten and restore a fixed set of Flax NNX module states."""

    def __init__(self, **modules: Any) -> None:
        self.names = tuple(modules)
        self._name_indices = {name: index for index, name in enumerate(self.names)}
        self._graphdefs = []
        self._treedefs = []
        self._leaves = []
        for name in self.names:
            graphdef, state = nnx.split(modules[name])
            leaves, treedef = jax.tree_util.tree_flatten(state)
            self._graphdefs.append(graphdef)
            self._treedefs.append(treedef)
            self._leaves.append(tuple(leaves))
        self._leaf_counts = tuple(len(leaves) for leaves in self._leaves)

    @classmethod
    def from_abstract_state(cls, abstract_state: Any) -> NnxStatePacker:
        instance = cls.__new__(cls)
        modules = {
            "model": nnx.merge(
                abstract_state.module_graphdef, abstract_state.module_state
            ),
            "optimizer": nnx.merge(
                abstract_state.optimizer_graphdef, abstract_state.optimizer_state
            ),
        }
        cls.__init__(instance, **modules)
        return instance

    def get_flat_leaves(self) -> tuple:
        return tuple(leaf for leaves in self._leaves for leaf in leaves)

    def leaf_count(self, name: str) -> int:
        try:
            return self._leaf_counts[self._name_indices[name]]
        except KeyError as error:
            raise KeyError(f"Unknown NnxStatePacker group: {name}") from error

    def pack(self, **objects: Any) -> tuple:
        return self.repack(**objects)

    def unflatten(self, flat_leaves: tuple) -> dict[str, Any]:
        states = self.unflatten_states(flat_leaves)
        return {
            name: nnx.merge(self._graphdefs[index], states[name])
            for index, name in enumerate(self.names)
        }

    def unflatten_group(self, name: str, flat_leaves: tuple) -> Any:
        try:
            index = self._name_indices[name]
        except KeyError as error:
            raise KeyError(f"Unknown NnxStatePacker group: {name}") from error
        return nnx.merge(
            self._graphdefs[index], self.unflatten_state(name, flat_leaves)
        )

    def unflatten_state(self, name: str, flat_leaves: tuple) -> Any:
        try:
            index = self._name_indices[name]
        except KeyError as error:
            raise KeyError(f"Unknown NnxStatePacker group: {name}") from error
        expected = self._leaf_counts[index]
        if len(flat_leaves) != expected:
            raise ValueError(
                f"NnxStatePacker group {name!r} requires {expected} leaves, "
                f"got {len(flat_leaves)}"
            )
        return self._treedefs[index].unflatten(flat_leaves)

    def unflatten_states(self, flat_leaves: tuple) -> dict[str, Any]:
        cursor = 0
        restored = {}
        for index, name in enumerate(self.names):
            count = self._leaf_counts[index]
            current = flat_leaves[cursor : cursor + count]
            cursor += count
            restored[name] = self._treedefs[index].unflatten(current)
        if cursor != len(flat_leaves):
            raise ValueError(
                f"NnxStatePacker received {len(flat_leaves)} leaves, expected {cursor}"
            )
        return restored

    def iter_state_leaves(self, **states: Any):
        for name in self.names:
            state_key = f"{name}_state"
            if state_key in states:
                state_obj = states[state_key]
            elif name in states:
                state_obj = states[name]
            else:
                raise KeyError(
                    f"Missing state for NnxStatePacker group {name!r}; "
                    f"expected {state_key!r} or {name!r}"
                )
            leaves, treedef = jax.tree_util.tree_flatten(state_obj)
            if treedef != self._treedefs[self._name_indices[name]]:
                raise ValueError(
                    f"Tree structure for state group {name!r} does not match packer definition"
                )
            yield from leaves

    def repack(self, **restored_objects: Any) -> tuple:
        return tuple(self.iter_repack(**restored_objects))

    def iter_repack(self, **restored_objects: Any):
        """Yield restored state leaves without retaining a second flat tuple."""
        for name in self.names:
            if name not in restored_objects:
                raise ValueError(f"Missing object for NnxStatePacker.repack: {name}")
            _, state = nnx.split(restored_objects[name])
            current, _ = jax.tree_util.tree_flatten(state)
            yield from current


class DictPacker:
    """Pack mappings into a stable tuple order across a JIT boundary."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = tuple(keys)
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("DictPacker keys must be unique")

    def pack(self, data: Mapping[str, object]) -> tuple:
        return tuple(data[key] for key in self.keys)

    def unpack(self, values: tuple) -> dict[str, object]:
        if len(values) != len(self.keys):
            raise ValueError(
                f"DictPacker received {len(values)} values, expected {len(self.keys)}"
            )
        return dict(zip(self.keys, values, strict=True))


__all__ = [
    "DictPacker",
    "NnxStatePacker",
]
