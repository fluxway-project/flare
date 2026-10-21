from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.state.layout import FlatStateLayout
from flare.jax.training.state.packers import NnxStatePacker


def _path_string(path: tuple[Any, ...]) -> str:
    return jtu.keystr(path)


def _matching_parameter_paths(
    optimizer_path: str, parameter_paths: frozenset[str]
) -> tuple[str, ...]:
    return tuple(
        parameter_path
        for parameter_path in parameter_paths
        if optimizer_path.endswith(parameter_path)
    )


@dataclass(frozen=True, slots=True)
class _OptimizerLeafMapping:
    optimizer_path: str
    shape: tuple[int, ...]
    dtype: jnp.dtype
    matched_parameter_paths: tuple[str, ...]

    @property
    def is_scalar(self) -> bool:
        return self.shape == ()

    @property
    def is_ambiguous(self) -> bool:
        return len(self.matched_parameter_paths) > 1

    @property
    def is_unmapped(self) -> bool:
        return len(self.matched_parameter_paths) == 0

    @property
    def parameter_path(self) -> str | None:
        if len(self.matched_parameter_paths) == 1:
            (param_path,) = self.matched_parameter_paths
            return param_path
        return None


def _tree_entries(tree: Any) -> list[tuple[Any, Any]]:
    entries, _treedef = jtu.tree_flatten_with_path(tree)
    return entries


@dataclass(frozen=True, slots=True)
class NnxShardUpdatePlan:
    """Canonical ownership metadata for a sharded NNX optimizer transition.

    The plan is derived from the model definition and optimizer state tree once,
    before tracing. Runtime update code can therefore use stable leaf indices and
    never infer ownership from positional coincidence or tensor shape alone.
    """

    parameter_paths: frozenset[str]
    scalar_parameter_paths: frozenset[str]
    unsupported_optimizer_paths: frozenset[str]
    incompatible_optimizer_paths: frozenset[str]
    optimizer_capability: str | None
    optimizer_ownership_contract: str | None
    model_parameter_indices: tuple[int, ...]
    gradient_parameter_indices: tuple[int, ...]

    @classmethod
    def from_definition(
        cls, definition: NnxTrainingStateDefinition
    ) -> NnxShardUpdatePlan:
        if not isinstance(definition, NnxTrainingStateDefinition):
            raise TypeError(
                "definition must be NnxTrainingStateDefinition, "
                f"got {type(definition).__name__}"
            )
        abstract = definition.abstract_state()
        parameter_paths = frozenset(definition.model_definition.param_paths)
        optimizer_entries = _tree_entries(abstract.optimizer_state)
        capability = definition.optimizer_shard_update_capability
        model_entries = _tree_entries(abstract.module_state)
        parameter_specs = {
            _path_string(path): (tuple(leaf.shape), jnp.dtype(leaf.dtype))
            for path, leaf in model_entries
            if _path_string(path) in parameter_paths
        }
        missing_parameter_metadata = parameter_paths.difference(parameter_specs)
        if missing_parameter_metadata:
            raise ValueError(
                "NNX model definition Param paths are absent from abstract model "
                f"state metadata: {sorted(missing_parameter_metadata)}"
            )
        scalar_parameter_paths = frozenset(
            path for path, (shape, _dtype) in parameter_specs.items() if not shape
        )
        leaf_mappings = tuple(
            _OptimizerLeafMapping(
                optimizer_path=_path_string(path),
                shape=tuple(leaf.shape),
                dtype=jnp.dtype(leaf.dtype),
                matched_parameter_paths=_matching_parameter_paths(
                    _path_string(path), parameter_paths
                ),
            )
            for path, leaf in optimizer_entries
        )
        unsupported_optimizer_paths = frozenset(
            mapping.optimizer_path
            for mapping in leaf_mappings
            if not mapping.is_scalar and (mapping.is_unmapped or mapping.is_ambiguous)
        )
        incompatible_optimizer_paths = frozenset(
            mapping.optimizer_path
            for mapping in leaf_mappings
            if mapping.parameter_path is not None
            and (mapping.shape, mapping.dtype)
            != parameter_specs[mapping.parameter_path]
        )
        model_parameter_indices = tuple(
            index
            for index, (path, _) in enumerate(model_entries)
            if _path_string(path) in parameter_paths
        )
        gradient_entries = _tree_entries(definition.abstract_gradient_state())
        gradient_paths = frozenset(_path_string(path) for path, _ in gradient_entries)
        if gradient_paths != parameter_paths:
            raise ValueError(
                "NNX gradient Param paths must exactly match model Param paths; "
                f"missing={sorted(parameter_paths - gradient_paths)}, "
                f"extra={sorted(gradient_paths - parameter_paths)}"
            )
        gradient_specs = {
            _path_string(path): (tuple(leaf.shape), jnp.dtype(leaf.dtype))
            for path, leaf in gradient_entries
        }
        incompatible_gradient_paths = frozenset(
            path
            for path in parameter_paths
            if gradient_specs[path] != parameter_specs[path]
        )
        if incompatible_gradient_paths:
            raise ValueError(
                "NNX gradient leaves must match their model Param shape and dtype; "
                f"incompatible paths={sorted(incompatible_gradient_paths)}"
            )
        gradient_parameter_indices = tuple(
            index
            for index, (path, _) in enumerate(gradient_entries)
            if _path_string(path) in parameter_paths
        )
        return cls(
            parameter_paths=parameter_paths,
            scalar_parameter_paths=scalar_parameter_paths,
            unsupported_optimizer_paths=unsupported_optimizer_paths,
            incompatible_optimizer_paths=incompatible_optimizer_paths,
            optimizer_capability=None if capability is None else capability.name,
            optimizer_ownership_contract=(
                None if capability is None else capability.ownership_contract
            ),
            model_parameter_indices=model_parameter_indices,
            gradient_parameter_indices=gradient_parameter_indices,
        )

    def validate_shardable(self) -> None:
        if self.optimizer_capability is None:
            raise ValueError(
                "sharded NNX optimizer update requires an explicit optimizer "
                "capability declaration"
            )
        if self.scalar_parameter_paths:
            raise ValueError(
                "sharded NNX optimizer update cannot partition scalar Param leaves; "
                f"unsupported paths={sorted(self.scalar_parameter_paths)}"
            )
        if self.optimizer_ownership_contract != "per_param_elementwise":
            raise ValueError(
                "sharded NNX optimizer update requires the per_param_elementwise "
                "ownership contract"
            )
        if self.unsupported_optimizer_paths:
            raise ValueError(
                "sharded NNX optimizer update requires every non-scalar optimizer "
                "leaf to be owned by a canonical Param path; unsupported paths="
                f"{sorted(self.unsupported_optimizer_paths)}"
            )
        if self.incompatible_optimizer_paths:
            raise ValueError(
                "sharded NNX optimizer update requires optimizer state leaves to "
                "match their Param shape and dtype; incompatible paths="
                f"{sorted(self.incompatible_optimizer_paths)}"
            )

    def build_runtime(
        self,
        *,
        definition: NnxTrainingStateDefinition,
        packer: NnxStatePacker,
        state_layout: FlatStateLayout,
        gradient_layout: FlatStateLayout,
        gradient_treedef: Any,
        world_size: int,
        axis_name: str,
    ) -> NnxShardUpdateRuntime:
        if world_size <= 1:
            raise ValueError("sharded update requires world_size greater than one")
        if not axis_name:
            raise ValueError("axis_name must be non-empty")
        if not isinstance(packer, NnxStatePacker):
            raise TypeError(
                f"packer must be NnxStatePacker, got {type(packer).__name__}"
            )
        self.validate_shardable()
        if len(self.model_parameter_indices) != len(self.gradient_parameter_indices):
            raise ValueError("model and gradient Param leaf counts must match")
        return NnxShardUpdateRuntime(
            plan=self,
            definition=definition,
            packer=packer,
            state_layout=state_layout,
            gradient_layout=gradient_layout,
            gradient_treedef=gradient_treedef,
            model_leaf_count=packer.leaf_count("model"),
            world_size=world_size,
            axis_name=axis_name,
        )


@dataclass(frozen=True, slots=True)
class NnxShardUpdateRuntime:
    """JIT-safe sharded state transformation around the canonical NNX update."""

    plan: NnxShardUpdatePlan
    definition: NnxTrainingStateDefinition
    packer: NnxStatePacker
    state_layout: FlatStateLayout
    gradient_layout: FlatStateLayout
    gradient_treedef: Any
    model_leaf_count: int
    world_size: int
    axis_name: str

    def _shard_slice(self, value: Any, local_size: int) -> Any:
        flat = jnp.reshape(value, (-1,))
        return jnp.reshape(
            flat.reshape(self.world_size, local_size)[
                jax.lax.axis_index(self.axis_name)
            ],
            (local_size,),
        )

    def materialize_shard_owned_state(
        self, flat_state: tuple[Any, ...]
    ) -> dict[str, Any]:
        if len(flat_state) != len(self.state_layout.leaves):
            raise ValueError("flat state count does not match sharded update layout")
        parameter_indices = frozenset(self.plan.model_parameter_indices)
        local_leaves: list[Any] = []
        for index, (value, layout) in enumerate(
            zip(flat_state, self.state_layout.leaves, strict=True)
        ):
            if index < self.model_leaf_count and index in parameter_indices:
                local_leaves.append(self._shard_slice(value, layout.local_size))
            elif index < self.model_leaf_count:
                local_leaves.append(layout.from_padded_flat(value))
            elif layout.original_size == 1:
                local_leaves.append(jnp.reshape(value, ()))
            else:
                local_leaves.append(jnp.reshape(value, (layout.local_size,)))
        return self.packer.unflatten(tuple(local_leaves))

    def materialize_shard_owned_gradients(
        self, gradients: tuple[Any, ...], *, replicated: bool
    ) -> Any:
        if len(gradients) != len(self.gradient_layout.leaves):
            raise ValueError("gradient count does not match sharded update layout")
        local_gradient_leaves: list[Any] = []
        for value, layout in zip(gradients, self.gradient_layout.leaves, strict=True):
            if replicated:
                value = self._shard_slice(value, layout.local_size)
            else:
                value = jnp.reshape(value, (layout.local_size,))
            local_gradient_leaves.append(value)
        return self.gradient_treedef.unflatten(local_gradient_leaves)

    def publish_replicated_model_and_sharded_optimizer(
        self, objects: dict[str, Any]
    ) -> tuple[Any, ...]:
        local_flat = self.packer.repack(**objects)
        parameter_indices = frozenset(self.plan.model_parameter_indices)
        published_state_leaves: list[Any] = []
        for index, (value, layout) in enumerate(
            zip(local_flat, self.state_layout.leaves, strict=True)
        ):
            if index < self.model_leaf_count and index in parameter_indices:
                local = jnp.reshape(value, (layout.local_size,))
                published_state_leaves.append(
                    jax.lax.all_gather(local, self.axis_name, axis=0, tiled=True)
                )
            elif index < self.model_leaf_count:
                published_state_leaves.append(layout.to_padded_flat(value))
            elif layout.original_size == 1:
                published_state_leaves.append(jnp.reshape(value, (layout.local_size,)))
            else:
                published_state_leaves.append(jnp.reshape(value, (layout.local_size,)))
        return tuple(published_state_leaves)


__all__ = ["NnxShardUpdatePlan", "NnxShardUpdateRuntime"]
