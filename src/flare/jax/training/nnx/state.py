from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.layers.state import NnxModelDefinition
from flare.jax.mesh import ParallelMesh, state_with_sharding
from flare.jax.roles import ConstantBuffer, Fp32Param, cast_floating_tree
from flare.jax.training.optimizers import ShardUpdateCapability
from flare.jax.training.precision import PrecisionPolicy
from flare.jax.training.state.packers import NnxStatePacker
from flare.jax.training.strategy import (
    CheckpointShardingStrategy,
    ZeroShardingStrategy,
)
from flare.training.state import TrainingStateBlueprint, TrainingStateDefinition
from flare.training.zero.contracts import ZeroStage


@dataclass(frozen=True, slots=True)
class NnxAbstractState:
    module_graphdef: Any
    module_state: Any
    optimizer_graphdef: Any
    optimizer_state: Any


def _path_string(path: tuple[Any, ...]) -> str:
    return jtu.keystr(path)


def _state_leaf_metadata(
    name: str, state: Any, role: str
) -> tuple[tuple[str, tuple[int, ...], str, str], ...]:
    leaves = []
    for path, leaf in jtu.tree_flatten_with_path(state)[0]:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError(
                f"abstract NNX state leaf {name}/{_path_string(path)!r} lacks shape/dtype"
            )
        leaves.append(
            (
                f"{name}/{_path_string(path)}",
                tuple(int(dimension) for dimension in leaf.shape),
                str(leaf.dtype),
                role,
            )
        )
    return tuple(leaves)


def _state_variable_types(state: Any) -> tuple[str, ...]:
    if not isinstance(state, nnx.State):
        raise TypeError(f"NNX state must be flax.nnx.State, got {type(state).__name__}")
    return tuple(
        f"{variable.type.__module__}.{variable.type.__qualname__}"
        for _, variable in nnx.to_flat_state(state)
    )


_ALLOWED_PARAM_DTYPES = frozenset(
    {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)}
)


def _raise_if_constant_changed(equal: Any) -> None:
    if not bool(equal):
        raise RuntimeError("ConstantBuffer state changed during forward computation")


class NnxTrainingStateDefinition(TrainingStateDefinition):
    def __init__(
        self,
        model_definition: NnxModelDefinition,
        optimizer_initializer: Callable[[nnx.Module], nnx.Optimizer],
        *,
        precision_policy: PrecisionPolicy = PrecisionPolicy.FP32,
    ) -> None:
        if not isinstance(model_definition, NnxModelDefinition):
            raise TypeError("model_definition must be NnxModelDefinition")
        if not callable(optimizer_initializer):
            raise TypeError("optimizer_initializer must be callable")
        if not isinstance(precision_policy, PrecisionPolicy):
            raise TypeError(
                "precision_policy must be PrecisionPolicy, "
                f"got {type(precision_policy).__name__}"
            )
        self._model_definition = model_definition
        self._optimizer_initializer = optimizer_initializer
        self._precision_policy = precision_policy

        module_graphdef = model_definition.graphdef
        module_state = model_definition.abstract_model_state()
        abstract_module = nnx.merge(module_graphdef, module_state)
        abstract_optimizer = nnx.eval_shape(
            lambda: optimizer_initializer(abstract_module)
        )
        if not isinstance(abstract_optimizer, nnx.Optimizer):
            raise TypeError("optimizer initializer must return flax.nnx.Optimizer")
        capability = getattr(abstract_optimizer, "_flare_shard_update_capability", None)
        if capability is not None and not isinstance(capability, ShardUpdateCapability):
            raise TypeError(
                "_flare_shard_update_capability must be ShardUpdateCapability"
            )
        self._optimizer_shard_update_capability = capability
        optimizer_graphdef, optimizer_state = nnx.split(abstract_optimizer)
        self._validate_optimizer_state(optimizer_state)

        self._module_graphdef = module_graphdef
        self._param_paths = model_definition.param_paths
        _, _, _, self._abstract_mutable_non_param_state = nnx.split(
            abstract_module, nnx.Param, ConstantBuffer, ...
        )
        leaves = _state_leaf_metadata("model", module_state, "model")
        leaves += _state_leaf_metadata("optimizer", optimizer_state, "optimizer")
        digest_payload = json.dumps(
            {
                "leaves": leaves,
                "variable_types": {
                    "model": _state_variable_types(module_state),
                    "optimizer": _state_variable_types(optimizer_state),
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self._blueprint = TrainingStateBlueprint(
            groups=("model", "optimizer"),
            leaves=leaves,
            schema_digest=hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
        )
        self._abstract = NnxAbstractState(
            module_graphdef=module_graphdef,
            module_state=module_state,
            optimizer_graphdef=optimizer_graphdef,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def from_factory(
        cls,
        factory: Any,
        *,
        precision_policy: PrecisionPolicy = PrecisionPolicy.FP32,
    ) -> NnxTrainingStateDefinition:
        if not hasattr(factory, "create_module") or not hasattr(
            factory, "create_optimizer"
        ):
            raise TypeError("factory must expose create_module and create_optimizer")
        return cls(
            NnxModelDefinition.from_factory(factory),
            factory.create_optimizer,
            precision_policy=precision_policy,
        )

    @classmethod
    def from_model_definition(
        cls,
        model_definition: NnxModelDefinition,
        optimizer_initializer: Callable[[nnx.Module], nnx.Optimizer],
        *,
        precision_policy: PrecisionPolicy = PrecisionPolicy.FP32,
    ) -> NnxTrainingStateDefinition:
        return cls(
            model_definition,
            optimizer_initializer,
            precision_policy=precision_policy,
        )

    @property
    def precision_policy(self) -> PrecisionPolicy:
        return self._precision_policy

    @property
    def optimizer_shard_update_capability(self) -> ShardUpdateCapability | None:
        """Explicit proof token required by ZeRO-1/2 sharded optimizer updates."""
        return self._optimizer_shard_update_capability

    @property
    def blueprint(self) -> TrainingStateBlueprint:
        return self._blueprint

    def abstract_state(self) -> NnxAbstractState:
        return self._abstract

    def abstract_model_state(self) -> Any:
        return self._abstract.module_state

    @property
    def model_definition(self) -> NnxModelDefinition:
        return self._model_definition

    def initialize_values(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        module = self._model_definition.initialize(*args, **kwargs)
        optimizer = self._optimizer_initializer(module)
        if not isinstance(optimizer, nnx.Optimizer):
            raise TypeError("optimizer initializer must return flax.nnx.Optimizer")
        self._validate_optimizer_state(nnx.split(optimizer)[1])
        return {"model": module, "optimizer": optimizer}

    def initialize_from_model_state(self, model_state: Any) -> Mapping[str, Any]:
        """Load external model weights through the current abstract graphdef."""
        module = self.materialize_model_from_state(model_state)
        optimizer = self._optimizer_initializer(module)
        if not isinstance(optimizer, nnx.Optimizer):
            raise TypeError("optimizer initializer must return flax.nnx.Optimizer")
        self._validate_optimizer_state(nnx.split(optimizer)[1])
        values = {"model": module, "optimizer": optimizer}
        self.validate_values(values)
        return values

    def materialize_model_from_state(self, model_state: Any) -> nnx.Module:
        """Materialize only a model from external weights for inference."""
        return self._model_definition.materialize(
            model_state,
            allowed_param_dtypes=self._allowed_param_dtypes(),
        )

    def initialize_from_model_loader(
        self, loader: Callable[[TrainingStateBlueprint], Any]
    ) -> Mapping[str, Any]:
        if not callable(loader):
            raise TypeError("model state loader must be callable")
        return self.initialize_from_model_state(loader(self.blueprint))

    def materialize_model_from_loader(
        self, loader: Callable[[TrainingStateBlueprint], Any]
    ) -> nnx.Module:
        if not callable(loader):
            raise TypeError("model state loader must be callable")
        return self.materialize_model_from_state(loader(self.blueprint))

    def validate_values(self, values: Mapping[str, Any]) -> None:
        if set(values) != {"model", "optimizer"}:
            raise KeyError("NNX training values must contain model and optimizer")
        _, model_state = nnx.split(values["model"])
        _, optimizer_state = nnx.split(values["optimizer"])
        self.validate_checkpoint_state(model_state, optimizer_state)

    def validate_checkpoint_state(self, model_state: Any, optimizer_state: Any) -> None:
        self._validate_state_tree("model", self._abstract.module_state, model_state)
        self._validate_state_tree(
            "optimizer", self._abstract.optimizer_state, optimizer_state
        )
        self._validate_optimizer_state(optimizer_state)

    def checkpoint_layout(
        self,
        sharding: NamedSharding | CheckpointShardingStrategy | ParallelMesh,
        *,
        axis_name: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(sharding, CheckpointShardingStrategy):
            cp_strat = sharding
            mesh = cp_strat.mesh
        elif isinstance(sharding, ParallelMesh):
            resolved_axis = (
                axis_name if axis_name is not None else sharding.axis_names[0]
            )
            cp_strat = CheckpointShardingStrategy(sharding, resolved_axis)
            mesh = sharding
        elif isinstance(sharding, NamedSharding):
            step_sharding = NamedSharding(sharding.mesh, P())
            return {
                "step": jax.ShapeDtypeStruct((), jnp.int32, sharding=step_sharding),
                "model_state": state_with_sharding(
                    self._abstract.module_state, sharding
                ),
                "optimizer_state": state_with_sharding(
                    self._abstract.optimizer_state, sharding
                ),
            }
        else:
            raise TypeError(
                "sharding must be NamedSharding, CheckpointShardingStrategy, or "
                f"ParallelMesh, got {type(sharding).__name__}"
            )

        return {
            "step": jax.ShapeDtypeStruct((), jnp.int32, sharding=mesh.replicated),
            "model_state": jtu.tree_map(
                lambda leaf: jax.ShapeDtypeStruct(
                    leaf.shape,
                    leaf.dtype,
                    sharding=cp_strat.checkpoint_sharding(leaf.shape),
                ),
                self._abstract.module_state,
            ),
            "optimizer_state": jtu.tree_map(
                lambda leaf: jax.ShapeDtypeStruct(
                    leaf.shape,
                    leaf.dtype,
                    sharding=cp_strat.checkpoint_sharding(leaf.shape),
                ),
                self._abstract.optimizer_state,
            ),
        }

    def shard_model(
        self,
        strategy: ZeroShardingStrategy | ParallelMesh,
        *,
        stage: ZeroStage = ZeroStage.REPLICATED,
        axis_name: str | None = None,
    ) -> Any:
        """Assign ZeRO parameter sharding to the abstract model state."""
        if isinstance(strategy, ParallelMesh):
            resolved_axis = (
                axis_name if axis_name is not None else strategy.axis_names[0]
            )
            zero_strategy = ZeroShardingStrategy(
                strategy, stage=stage, axis_name=resolved_axis
            )
        elif isinstance(strategy, ZeroShardingStrategy):
            zero_strategy = strategy
        else:
            raise TypeError(
                "strategy must be ZeroShardingStrategy or ParallelMesh, "
                f"got {type(strategy).__name__}"
            )
        return zero_strategy.shard_model(self._abstract.module_state)

    def shard_optimizer(
        self,
        strategy: ZeroShardingStrategy | ParallelMesh,
        *,
        stage: ZeroStage = ZeroStage.REPLICATED,
        axis_name: str | None = None,
    ) -> Any:
        """Assign ZeRO optimizer state sharding to the abstract optimizer state."""
        if isinstance(strategy, ParallelMesh):
            resolved_axis = (
                axis_name if axis_name is not None else strategy.axis_names[0]
            )
            zero_strategy = ZeroShardingStrategy(
                strategy, stage=stage, axis_name=resolved_axis
            )
        elif isinstance(strategy, ZeroShardingStrategy):
            zero_strategy = strategy
        else:
            raise TypeError(
                "strategy must be ZeroShardingStrategy or ParallelMesh, "
                f"got {type(strategy).__name__}"
            )
        return zero_strategy.shard_optimizer(self._abstract.optimizer_state)

    def validate_checkpoint_payload(
        self, state: Mapping[str, Any], context: str = "checkpoint"
    ) -> int:
        required = {"step", "model_state", "optimizer_state"}
        missing = required.difference(state)
        if missing:
            raise KeyError(f"{context} missing fields: {sorted(missing)}")
        unexpected = set(state).difference(required)
        if unexpected:
            raise ValueError(
                f"{context} contains unsupported fields: {sorted(unexpected)}"
            )
        self.validate_checkpoint_state(state["model_state"], state["optimizer_state"])
        return self._read_checkpoint_step(state["step"])

    @staticmethod
    def _read_checkpoint_step(value: Any) -> int:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError("checkpoint step must expose shape and dtype")
        if tuple(shape) != ():
            raise ValueError(
                f"checkpoint step must be scalar, got shape {tuple(shape)}"
            )
        if jnp.dtype(dtype) != jnp.dtype(jnp.int32):
            raise TypeError(f"checkpoint step must be int32, got {jnp.dtype(dtype)}")
        step = int(np.asarray(jax.device_get(value)).item())
        if step < 0:
            raise ValueError("checkpoint step must be non-negative")
        return step

    def validate_compute_module(self, module: nnx.Module) -> None:
        """Validate the static model contract at an explicit preflight boundary."""
        if not isinstance(module, nnx.Module):
            raise TypeError(
                f"module must be flax.nnx.Module, got {type(module).__name__}"
            )
        graphdef, state = nnx.split(module)
        if graphdef != self._abstract.module_graphdef:
            raise ValueError("NNX module graph differs from the model definition")
        self._model_definition.validate_state(
            state, allowed_param_dtypes=self._allowed_param_dtypes()
        )

    def prepare_compute_module(self, module: nnx.Module) -> nnx.Module:
        """Create a compute view using the already-validated model contract."""
        if not isinstance(module, nnx.Module):
            raise TypeError(
                f"module must be flax.nnx.Module, got {type(module).__name__}"
            )
        if self.precision_policy is PrecisionPolicy.FP32:
            return module
        graphdef, _ = nnx.split(module)
        _, fp32_param_state, param_state, constant_state, mutable_non_param_state = (
            nnx.split(module, Fp32Param, nnx.Param, ConstantBuffer, ...)
        )
        compute_params = jtu.tree_map(
            lambda leaf: self._cast_array(leaf, self.precision_policy.compute_dtype),
            param_state,
        )
        compute_non_param_state = jtu.tree_map(
            self._copy_array, mutable_non_param_state
        )
        compute_module = nnx.merge(
            graphdef,
            fp32_param_state,
            compute_params,
            jtu.tree_map(self._copy_array, constant_state),
            compute_non_param_state,
        )
        if not isinstance(compute_module, nnx.Module):
            raise TypeError("NNX compute state did not materialize an NNX module")
        return compute_module

    def prepare_compute_module_checked(self, module: nnx.Module) -> nnx.Module:
        """Validate static state once, then create the compute view."""
        self.validate_compute_module(module)
        return self.prepare_compute_module(module)

    def sync_non_param_state(
        self,
        master_module: nnx.Module,
        compute_module: nnx.Module,
        *,
        validate_constant_values: bool = False,
    ) -> None:
        """Copy mutable non-Param state without host work in the default path.

        ConstantBuffer schemas are validated when the definition is built. Value
        comparison is an opt-in diagnostic because a traced comparison otherwise
        inserts a host callback into every low-precision micro-step.
        """
        if not isinstance(master_module, nnx.Module) or not isinstance(
            compute_module, nnx.Module
        ):
            raise TypeError("master_module and compute_module must be NNX modules")
        if master_module is compute_module:
            return
        _, _, compute_constants, compute_non_param_state = nnx.split(
            compute_module, nnx.Param, ConstantBuffer, ...
        )
        if validate_constant_values:
            master_graphdef, _ = nnx.split(master_module)
            compute_graphdef, _ = nnx.split(compute_module)
            if compute_graphdef != master_graphdef:
                raise ValueError("compute module graph differs from the master module")
            self.validate_compute_module(master_module)
            _, _, master_constants, _ = nnx.split(
                master_module, nnx.Param, ConstantBuffer, ...
            )
            master_constant_leaves = jtu.tree_leaves(master_constants)
            compute_constant_leaves = jtu.tree_leaves(compute_constants)
            if len(master_constant_leaves) != len(compute_constant_leaves):
                raise RuntimeError("ConstantBuffer state schema changed")
            constant_equal = jnp.asarray(True)
            for master_leaf, compute_leaf in zip(
                master_constant_leaves,
                compute_constant_leaves,
                strict=True,
            ):
                if isinstance(master_leaf, jax.ShapeDtypeStruct) or isinstance(
                    compute_leaf, jax.ShapeDtypeStruct
                ):
                    if tuple(master_leaf.shape) != tuple(
                        compute_leaf.shape
                    ) or jnp.dtype(master_leaf.dtype) != jnp.dtype(compute_leaf.dtype):
                        raise RuntimeError("ConstantBuffer state schema changed")
                else:
                    constant_equal = jnp.logical_and(
                        constant_equal, jnp.array_equal(master_leaf, compute_leaf)
                    )
            if master_constant_leaves and isinstance(constant_equal, jax.core.Tracer):
                jax.debug.callback(_raise_if_constant_changed, constant_equal)
            elif master_constant_leaves and not bool(constant_equal):
                raise RuntimeError(
                    "ConstantBuffer state changed during forward computation"
                )
        nnx.update(master_module, compute_non_param_state)

    def prepare_inputs(self, inputs: Any) -> Any:
        if not isinstance(inputs, Mapping):
            raise TypeError(f"inputs must be a mapping, got {type(inputs).__name__}")
        return cast_floating_tree(inputs, self.precision_policy.compute_dtype)

    def wrap_loss_fn(
        self, loss_fn: Callable[[Any, Mapping[str, Any]], Any]
    ) -> Callable[[Any, Mapping[str, Any]], Any]:
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")

        def loss_with_fp32_boundary(
            predictions: Any, targets: Mapping[str, Any]
        ) -> Any:
            fp32_predictions = cast_floating_tree(predictions, jnp.float32)
            fp32_targets = cast_floating_tree(targets, jnp.float32)
            loss = jnp.asarray(
                loss_fn(fp32_predictions, fp32_targets), dtype=jnp.float32
            )
            if loss.ndim != 0:
                raise ValueError(
                    f"loss_fn must return a scalar, got shape {loss.shape}"
                )
            return loss

        return loss_with_fp32_boundary

    def cast_predictions(self, predictions: Any) -> Any:
        return cast_floating_tree(predictions, jnp.float32)

    def cast_gradients(self, gradients: Any) -> Any:
        return jtu.tree_map(self._cast_gradient, gradients)

    def _allowed_param_dtypes(self) -> frozenset[jnp.dtype]:
        if self.precision_policy is PrecisionPolicy.FP32:
            return frozenset({jnp.dtype(self.precision_policy.parameter_dtype)})
        return _ALLOWED_PARAM_DTYPES

    def _validate_optimizer_state(self, optimizer_state: Any) -> None:
        expected_dtype = self.precision_policy.optimizer_dtype
        for path, leaf in jtu.tree_flatten_with_path(optimizer_state)[0]:
            if not hasattr(leaf, "dtype"):
                raise TypeError(
                    f"NNX optimizer state leaf {_path_string(path)!r} lacks dtype"
                )
            dtype = jnp.dtype(leaf.dtype)
            if jnp.issubdtype(dtype, jnp.inexact) and dtype != expected_dtype:
                raise TypeError(
                    f"NNX optimizer floating state must be {expected_dtype}, "
                    f"got {_path_string(path)!r}={dtype}"
                )

    @staticmethod
    def _cast_array(leaf: Any, dtype: jnp.dtype) -> Any:
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(
                leaf.shape, dtype, sharding=getattr(leaf, "sharding", None)
            )
        return jnp.asarray(leaf).astype(dtype)

    @staticmethod
    def _copy_array(leaf: Any) -> Any:
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return leaf
        return jnp.array(leaf, copy=True)

    @staticmethod
    def _cast_gradient(leaf: Any) -> Any:
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is None:
            return leaf
        if leaf_dtype == jax.dtypes.float0:
            shape = getattr(leaf, "shape", None)
            if shape is None:
                raise TypeError("float0 gradient must expose a shape")
            return jnp.zeros(shape, dtype=jnp.float32)
        source_dtype = jnp.dtype(leaf_dtype)
        if jnp.issubdtype(source_dtype, jnp.floating):
            return NnxTrainingStateDefinition._cast_array(leaf, jnp.float32)
        return leaf

    @staticmethod
    def _validate_state_tree(name: str, expected_state: Any, actual_state: Any) -> None:
        if jtu.tree_structure(expected_state) != jtu.tree_structure(actual_state):
            raise ValueError(f"NNX {name} state tree differs from blueprint")
        expected = jtu.tree_flatten_with_path(expected_state)[0]
        actual = jtu.tree_flatten_with_path(actual_state)[0]
        if len(expected) != len(actual):
            raise ValueError(
                f"NNX {name} state leaf count differs from abstract blueprint"
            )
        for (expected_path, expected_leaf), (actual_path, actual_leaf) in zip(
            expected, actual, strict=True
        ):
            if _path_string(expected_path) != _path_string(actual_path):
                raise ValueError(f"NNX {name} state tree differs from blueprint")
            if tuple(actual_leaf.shape) != tuple(expected_leaf.shape):
                raise ValueError(
                    f"NNX {name} leaf {_path_string(actual_path)!r} shape differs"
                )
            if str(actual_leaf.dtype) != str(expected_leaf.dtype):
                raise ValueError(
                    f"NNX {name} leaf {_path_string(actual_path)!r} dtype differs"
                )

    def packer(self) -> NnxStatePacker:
        return NnxStatePacker.from_abstract_state(self._abstract)

    def abstract_flat_leaves(self) -> tuple[Any, ...]:
        return self.packer().get_flat_leaves()

    def abstract_gradient_state(self) -> Any:
        module = nnx.merge(self._abstract.module_graphdef, self._abstract.module_state)
        _, params, _ = nnx.split(module, nnx.Param, ...)
        return jtu.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(
                leaf.shape, self.precision_policy.parameter_dtype
            ),
            params,
        )


__all__ = [
    "NnxAbstractState",
    "NnxTrainingStateDefinition",
]
