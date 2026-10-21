from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.mesh import ParallelMesh
from flare.jax.roles import ConstantBuffer
from flare.jax.training.engines.contracts import TrainStepResult
from flare.jax.training.gradients.input import (
    partition_input_gradients,
    validate_input_gradient_keys,
)
from flare.jax.training.gradients.normalization import normalize_input_gradients
from flare.jax.training.nnx.ownership import (
    NnxShardUpdatePlan,
    NnxShardUpdateRuntime,
)
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.state.layout import FlatStateLayout
from flare.jax.training.state.packers import NnxStatePacker
from flare.jax.training.steppers import (
    MacroUpdateOutput,
    build_shard_map_macro_update,
    build_shard_map_micro_accumulate,
    compute_module_and_input_gradients,
)
from flare.jax.training.strategy import CheckpointShardingStrategy, ZeroShardingStrategy
from flare.jax.training.zero import reduce_microstep_gradients
from flare.runtime.diagnostics.state_audit import fingerprint_state
from flare.training.execution import (
    DistributedTrainStepExecutor,
    OptimizerUpdateWindow,
    TrainingExecutionPlan,
)
from flare.training.state import TrainingStateLifecycle
from flare.training.zero.contracts import ZeroStage


def initialize_distributed_jax(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int | None,
    rank0_address: str | None,
) -> None:
    """Bootstrap JAX process topology before the backend can be initialized."""
    if world_size <= 1:
        raise ValueError(f"world_size must be greater than one, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if local_world_size is not None and local_world_size <= 0:
        raise ValueError(
            f"local_world_size must be positive when provided, got {local_world_size}"
        )
    rank_limit = world_size if local_world_size is None else local_world_size
    if local_rank < 0 or local_rank >= rank_limit:
        raise ValueError(f"local_rank must be in [0, {rank_limit}), got {local_rank}")
    if not rank0_address:
        raise ValueError("rank0_address is required for distributed JAX execution")
    if jax.distributed.is_initialized():
        if jax.process_index() != rank or jax.process_count() != world_size:
            raise RuntimeError(
                "JAX process topology was initialized with a different Fluxway worker topology: "
                f"jax=({jax.process_index()}, {jax.process_count()}) "
                f"worker=({rank}, {world_size})"
            )
        return
    # IMPORTANT RAY TOPOLOGY INVARIANT: each Ray worker process is launched
    # with exactly one visible GPU.  ``local_rank`` identifies the worker in
    # Fluxway's host topology; it is NOT an index into this process's visible
    # devices.  Keep ``local_device_ids=[0]``.  Using ``[local_rank]`` here
    # repeatedly breaks multi-worker startup by asking a one-GPU Ray process
    # to bind a device ordinal it cannot see.
    jax.distributed.initialize(
        coordinator_address=rank0_address,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0],
    )
    if jax.process_index() != rank or jax.process_count() != world_size:
        raise RuntimeError(
            "JAX process topology does not match Fluxway worker topology after initialization: "
            f"jax=({jax.process_index()}, {jax.process_count()}) "
            f"worker=({rank}, {world_size})"
        )


def _global_array_from_local(value: Any, *, mesh: Mesh, axis_name: str) -> jax.Array:
    local = jnp.asarray(value)
    return jax.make_array_from_single_device_arrays(
        (int(mesh.shape[axis_name]) * local.shape[0],) + tuple(local.shape[1:]),
        NamedSharding(mesh, P(axis_name)),
        [jax.device_put(local, device) for device in mesh.local_devices],
    )


def _replicated_array(value: Any, *, mesh: Mesh) -> jax.Array:
    local = jnp.asarray(value)
    return jax.make_array_from_single_device_arrays(
        local.shape,
        NamedSharding(mesh, P()),
        [jax.device_put(local, device) for device in mesh.local_devices],
    )


def _logical_checkpoint_leaf(
    value: jax.Array, *, original_size: int, original_shape: tuple[int, ...]
) -> jax.Array:
    return jnp.reshape(value[:original_size], original_shape)


def _require_replicated_state(equal: Any) -> None:
    if not bool(equal):
        raise RuntimeError(
            "distributed integer or boolean non-Param state differs across ranks"
        )


def _build_nnx_optimizer_update(
    *,
    stage: ZeroStage,
    packer: NnxStatePacker,
    state_layout: FlatStateLayout,
    gradient_layout: FlatStateLayout,
    gradient_treedef: Any,
    shard_update_runtime: NnxShardUpdateRuntime | None,
) -> Callable[[tuple[Any, ...], tuple[Any, ...], Any], Any]:
    def optimizer_update(
        flat_state: tuple[Any, ...],
        gradients: tuple[Any, ...],
        accumulation_steps: Any,
    ):
        normalized_gradients = jtu.tree_map(
            lambda value: value / accumulation_steps, gradients
        )
        if shard_update_runtime is not None:
            objects = shard_update_runtime.materialize_shard_owned_state(flat_state)
            gradient_state = shard_update_runtime.materialize_shard_owned_gradients(
                normalized_gradients,
                replicated=stage is ZeroStage.OPTIMIZER_STATE,
            )
            objects["optimizer"].update(objects["model"], gradient_state)
            return (
                shard_update_runtime.publish_replicated_model_and_sharded_optimizer(
                    objects
                ),
                normalized_gradients,
            )
        logical_state_leaves = state_layout.from_padded_leaves(flat_state)
        logical_gradient_leaves = gradient_layout.from_padded_leaves(
            normalized_gradients
        )
        objects = packer.unflatten(logical_state_leaves)
        module = objects["model"]
        optimizer = objects["optimizer"]
        gradient_state = gradient_treedef.unflatten(logical_gradient_leaves)
        optimizer.update(module, gradient_state)
        return (
            state_layout.to_padded_leaves(
                packer.repack(model=module, optimizer=optimizer)
            ),
            normalized_gradients,
        )

    return optimizer_update


class DistributedNnxTrainEngine:
    """Explicit multi-device NNX training engine with dense ZeRO semantics.

    State leaves are flattened and padded once.  ZeRO-1 keeps model leaves
    replicated and partitions optimizer leaves; ZeRO-2 additionally keeps the
    accumulated parameter gradients partitioned; ZeRO-3 partitions model and
    optimizer leaves and materializes model leaves with all-gather for each
    micro-step.  Optimizer updates run on globally-shaped arrays so NNX sees
    the original leaf shapes after decoding.
    """

    def __init__(
        self,
        *,
        state_definition: NnxTrainingStateDefinition,
        forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
        loss_fn: Callable[[Any, Mapping[str, Any]], Any],
        plan: TrainingExecutionPlan,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int | None,
        rank0_address: str | None,
        state_lifecycle: TrainingStateLifecycle = TrainingStateLifecycle.FRESH_INITIALIZE,
        model_state: Any | None = None,
        materialize_parameter_gradients: bool = False,
        input_gradient_keys: tuple[str, ...] | None = None,
        target_shardings: Mapping[str, P] | None = None,
        validate_non_param_replication: bool = False,
    ) -> None:
        if not jax.distributed.is_initialized():
            initialize_distributed_jax(
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                local_world_size=local_world_size,
                rank0_address=rank0_address,
            )
        if jax.process_index() != rank or jax.process_count() != world_size:
            raise RuntimeError(
                "JAX process topology does not match Fluxway worker topology: "
                f"jax=({jax.process_index()}, {jax.process_count()}) "
                f"worker=({rank}, {world_size})"
            )
        devices = tuple(jax.devices())
        if len(devices) != world_size:
            raise RuntimeError(
                "distributed JAX requires one visible device per worker: "
                f"devices={len(devices)}, world_size={world_size}"
            )
        if not isinstance(state_definition, NnxTrainingStateDefinition):
            raise TypeError(
                "state_definition must be NnxTrainingStateDefinition, "
                f"got {type(state_definition).__name__}"
            )
        self.plan = plan
        if not isinstance(state_lifecycle, TrainingStateLifecycle):
            raise TypeError(
                "state_lifecycle must be TrainingStateLifecycle, "
                f"got {type(state_lifecycle).__name__}"
            )
        self.state_lifecycle = state_lifecycle
        if not isinstance(materialize_parameter_gradients, bool):
            raise TypeError("materialize_parameter_gradients must be bool")
        if not isinstance(validate_non_param_replication, bool):
            raise TypeError("validate_non_param_replication must be bool")
        input_gradient_keys = validate_input_gradient_keys(input_gradient_keys)
        self._retain_parameter_gradients = materialize_parameter_gradients
        self._validate_non_param_replication = validate_non_param_replication
        self.input_gradient_keys = input_gradient_keys
        if not isinstance(target_shardings, Mapping) or not target_shardings:
            raise ValueError("target_shardings must be a non-empty mapping")
        if any(not isinstance(name, str) or not name for name in target_shardings):
            raise ValueError("target_shardings keys must be non-empty strings")
        if any(not isinstance(spec, P) for spec in target_shardings.values()):
            raise TypeError("target_shardings values must be PartitionSpec instances")
        self.target_keys = tuple(target_shardings)
        self.target_specs = tuple(target_shardings[key] for key in self.target_keys)
        if any(spec not in (P(), P("dp")) for spec in self.target_specs):
            raise ValueError(
                "distributed target shardings must be either replicated P() or row-sharded P('dp')"
            )
        self.state_definition = state_definition
        self.rank = rank
        self.world_size = world_size
        axis_name = plan.zero.axis_name
        resolved_devices = tuple(jax.devices() if devices is None else devices)
        self.parallel_mesh = ParallelMesh.from_axes(
            devices=resolved_devices, **{axis_name: len(resolved_devices)}
        )
        self.zero_strategy = ZeroShardingStrategy.from_execution_plan(
            plan, self.parallel_mesh
        )
        self.checkpoint_strategy = CheckpointShardingStrategy(
            self.parallel_mesh, self.axis_name
        )
        self.local_device = jax.local_devices()[0]
        self._mesh_devices_set = set(self.mesh.devices.flat)
        self.packer = state_definition.packer()
        self.model_leaf_count = self.packer.leaf_count("model")
        full_flat_state = state_definition.abstract_flat_leaves()
        self.state_layout = FlatStateLayout.from_abstract(full_flat_state, world_size)
        self.model_state_layout = self.state_layout.prefix(self.model_leaf_count)

        param_state = state_definition.abstract_gradient_state()
        self.gradient_treedef = jtu.tree_structure(param_state)
        self.gradient_layout = FlatStateLayout.from_abstract(
            tuple(jtu.tree_leaves(param_state)), world_size
        )
        self._checkpoint_value_materializers = (
            self._build_checkpoint_value_materializers(self.state_layout)
        )
        self.state_specs = self._state_specs()
        self._runtime_state_sharders_cache = self._build_runtime_state_sharders(
            self.state_layout, self.state_specs
        )
        self.flat_state: tuple[jax.Array, ...] | None = None
        self.gradient_specs = self.zero_strategy.zero_gradient_specs(
            len(self.gradient_layout.leaves)
        )
        self.accumulated_gradients: tuple[jax.Array, ...] | None = None
        self._last_padded_gradients: tuple[jax.Array, ...] | None = None
        self._last_raw_gradients: tuple[jax.Array, ...] | None = None
        self._last_reduced_gradients: tuple[jax.Array, ...] | None = None
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.input_keys: tuple[str, ...] | None = None
        self.micro_fn_jit: Callable[..., Any] | None = None
        self.micro_accumulate_fn_jit: Callable[..., Any] | None = None
        self.micro_accumulate_diagnostic_fn_jit: Callable[..., Any] | None = None
        self.macro_fn_jit: Callable[..., Any] | None = None
        self.step = 0
        self.micro_step = 0
        self._prepared_execution_signature: tuple[Any, ...] | None = None
        if self.state_lifecycle is TrainingStateLifecycle.FRESH_INITIALIZE:
            if model_state is not None:
                raise ValueError(
                    "model_state is only valid for the model-weights lifecycle"
                )
            self._ensure_concrete_state()
        elif self.state_lifecycle is TrainingStateLifecycle.MODEL_WEIGHTS:
            if model_state is None:
                raise ValueError(
                    "model_state is required for the model-weights lifecycle"
                )
            self._ensure_concrete_state(model_state=model_state)
        elif model_state is not None:
            raise ValueError(
                "model_state is only valid for the model-weights lifecycle"
            )

    @property
    def parallel_mesh(self) -> ParallelMesh:
        return self._parallel_mesh

    @parallel_mesh.setter
    def parallel_mesh(self, value: ParallelMesh) -> None:
        if not isinstance(value, ParallelMesh):
            raise TypeError(
                f"parallel_mesh must be ParallelMesh, got {type(value).__name__}"
            )
        self._parallel_mesh = value

    @property
    def mesh(self) -> Mesh:
        return self.parallel_mesh.mesh

    @property
    def axis_name(self) -> str:
        return self.parallel_mesh.axis_names[0]

    @property
    def _runtime_state_sharders(self) -> tuple[Callable[[Any], jax.Array], ...]:
        return self._runtime_state_sharders_cache

    def _build_runtime_state_sharders(
        self, layout: FlatStateLayout, specs: tuple[P, ...]
    ) -> tuple[Callable[[Any], jax.Array], ...]:
        input_shardings = tuple(NamedSharding(self.mesh, P()) for _ in layout.leaves)
        return self._build_state_sharders(layout, specs, input_shardings)

    def _build_checkpoint_state_sharders(
        self, layout: FlatStateLayout, specs: tuple[P, ...]
    ) -> tuple[Callable[[Any], jax.Array], ...]:
        input_shardings = tuple(
            self._checkpoint_sharding(leaf.original_shape) for leaf in layout.leaves
        )
        return self._build_state_sharders(layout, specs, input_shardings)

    def _build_state_sharders(
        self,
        layout: FlatStateLayout,
        specs: tuple[P, ...],
        input_shardings: tuple[NamedSharding, ...],
    ) -> tuple[Callable[[Any], jax.Array], ...]:
        if len(layout.leaves) != len(specs) or len(specs) != len(input_shardings):
            raise ValueError("state layout, output, and input sharding counts differ")
        return tuple(
            jax.jit(
                leaf.to_padded_flat,
                in_shardings=input_sharding,
                out_shardings=NamedSharding(self.mesh, spec),
            )
            for leaf, spec, input_sharding in zip(
                layout.leaves, specs, input_shardings, strict=True
            )
        )

    def _adapt_input_leaf(self, leaf: Any) -> Any:
        # Keep tensor on a single local device or host to avoid allocating redundant full
        # replicas across all devices in Python memory/VRAM prior to compiled encoding.
        if not isinstance(leaf, jax.Array):
            return leaf
        if getattr(getattr(leaf, "sharding", None), "spec", None) == P():
            return leaf
        mesh_devices = getattr(self, "_mesh_devices_set", None)
        if mesh_devices is None:
            mesh_devices = set(self.mesh.devices.flat)
            self._mesh_devices_set = mesh_devices
        leaf_devices = set(leaf.devices())
        if leaf_devices == mesh_devices:
            return leaf
        if len(leaf_devices) == 1:
            device = next(iter(leaf_devices))
            if device in self.mesh.local_devices:
                return leaf
        return jax.device_put(leaf, self.mesh.local_devices[0])

    def _ensure_concrete_state(self, *, model_state: Any | None = None) -> None:
        if self.flat_state is not None:
            return
        values = (
            self.state_definition.initialize_values()
            if model_state is None
            else self.state_definition.initialize_from_model_state(model_state)
        )
        self.state_definition.validate_values(values)
        self.state_definition.validate_compute_module(values["model"])
        self.flat_state = self._materialize_values(values)

    def _materialize_values(self, values: Mapping[str, Any]) -> tuple[jax.Array, ...]:
        materialized: list[jax.Array] = []
        values_iterator = self.packer.iter_repack(**values)
        for index, shard_state in enumerate(self._runtime_state_sharders):
            try:
                value = next(values_iterator)
            except StopIteration as error:
                raise ValueError(
                    "distributed state contains fewer leaves than its flat layout: "
                    f"index={index}, layout={len(self._runtime_state_sharders)}"
                ) from error
            adapted = self._adapt_input_leaf(value)
            materialized.append(shard_state(adapted))
            del adapted, value
        try:
            next(values_iterator)
        except StopIteration:
            return tuple(materialized)
        raise ValueError("distributed state contains more leaves than its flat layout")

    def _abstract_flat_layout(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        return tuple(
            jax.ShapeDtypeStruct(
                (layout.padded_size,),
                layout.dtype,
                sharding=NamedSharding(self.mesh, spec),
            )
            for layout, spec in zip(
                self.state_layout.leaves, self.state_specs, strict=True
            )
        )

    def _build_checkpoint_value_materializers(
        self, layout: FlatStateLayout
    ) -> tuple[Callable[[jax.Array], jax.Array], ...]:
        return tuple(
            jax.jit(
                leaf.from_padded_flat,
                out_shardings=self._checkpoint_sharding(leaf.original_shape),
            )
            for leaf in layout.leaves
        )

    def _checkpoint_sharding(self, shape: tuple[int, ...]) -> NamedSharding:
        return self.checkpoint_strategy.checkpoint_sharding(shape)

    def _global_checkpoint_values(
        self,
        values: tuple[jax.Array, ...],
        value_materializers: tuple[Callable[[jax.Array], jax.Array], ...],
    ) -> tuple[jax.Array, ...]:
        if len(values) != len(value_materializers):
            raise ValueError(
                "distributed checkpoint value and converter counts must match"
            )
        return tuple(
            materialize(value)
            for value, materialize in zip(values, value_materializers, strict=True)
        )

    def _abstract_gradient_layout(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        return tuple(
            jax.ShapeDtypeStruct(
                (layout.padded_size,),
                layout.dtype,
                sharding=NamedSharding(self.mesh, spec),
            )
            for layout, spec in zip(
                self.gradient_layout.leaves, self.gradient_specs, strict=True
            )
        )

    def _state_specs(self) -> tuple[P, ...]:
        return self.zero_strategy.zero_state_specs(
            self.model_leaf_count,
            len(self.state_layout.leaves),
        )

    def _build_functions(self, input_keys: tuple[str, ...]) -> None:
        if self.micro_fn_jit is not None:
            if self.input_keys != input_keys:
                raise ValueError(
                    "distributed input key order changed after JAX function construction: "
                    f"old={self.input_keys}, new={input_keys}"
                )
            return
        self.input_keys = input_keys
        if self.input_gradient_keys is not None:
            missing_keys = tuple(
                key for key in self.input_gradient_keys if key not in input_keys
            )
            if missing_keys:
                raise KeyError(
                    f"input gradient keys are missing from inputs: {missing_keys}"
                )
        stage = self.plan.zero.stage

        def micro_impl(
            flat_state: tuple[Any, ...],
            input_values: tuple[Any, ...],
            target_values: tuple[Any, ...],
        ):
            inputs = dict(zip(input_keys, input_values, strict=True))
            inputs = self.state_definition.prepare_inputs(inputs)
            module_inputs, differentiable_inputs, input_gradient_keys = (
                partition_input_gradients(inputs, self.input_gradient_keys)
            )
            targets = dict(zip(self.target_keys, target_values, strict=True))

            def model_from_flat(local_model: tuple[Any, ...]) -> nnx.Module:
                if stage is ZeroStage.PARAMETER:
                    model_values = self.model_state_layout.materialize_sharded_leaves(
                        local_model, axis_name=self.axis_name
                    )
                else:
                    model_values = self.model_state_layout.from_padded_leaves(
                        local_model
                    )
                return self.packer.unflatten_group("model", model_values)

            compute_loss = self.state_definition.wrap_loss_fn(self.loss_fn)
            module = model_from_flat(flat_state[: self.model_leaf_count])
            compute_module = self.state_definition.prepare_compute_module(module)
            (loss, predictions), (gradient_state, input_gradients) = (
                compute_module_and_input_gradients(
                    module=compute_module,
                    forward_fn=self.forward_fn,
                    loss_fn=compute_loss,
                    input_gradient_inputs=differentiable_inputs,
                    module_inputs=module_inputs,
                    targets=targets,
                )
            )
            gradient_state = self.state_definition.cast_gradients(gradient_state)
            input_gradients = self.state_definition.cast_gradients(input_gradients)
            predictions = self.state_definition.cast_predictions(predictions)
            self.state_definition.sync_non_param_state(module, compute_module)
            padded_gradients = tuple(
                value.astype(jnp.dtype(layout.dtype))
                for value, layout in zip(
                    jtu.tree_leaves(gradient_state),
                    self.gradient_layout.leaves,
                    strict=True,
                )
            )

            _, _, _, non_param_state = nnx.split(module, nnx.Param, ConstantBuffer, ...)

            def synchronize_non_param_leaf(leaf: Any) -> Any:
                dtype = jnp.dtype(leaf.dtype)
                if jnp.issubdtype(dtype, jnp.inexact):
                    return jax.lax.pmean(leaf, self.axis_name)
                if jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(
                    dtype, jnp.bool_
                ):
                    gathered = jax.lax.all_gather(
                        leaf, self.axis_name, axis=0, tiled=False
                    )
                    if self._validate_non_param_replication:
                        jax.debug.callback(
                            _require_replicated_state,
                            jnp.all(gathered == gathered[0]),
                        )
                    return gathered[0]
                raise TypeError(
                    "distributed non-Param state must be inexact, integer, or bool, "
                    f"got {dtype}"
                )

            nnx.update(
                module, jtu.tree_map(synchronize_non_param_leaf, non_param_state)
            )
            updated_model_state = nnx.split(module)[1]
            updated_model_values = self.model_state_layout.to_padded_leaves(
                tuple(jtu.tree_leaves(updated_model_state))
            )
            if stage is ZeroStage.PARAMETER:
                updated_model_values = tuple(
                    value.reshape(self.world_size, -1)[
                        jax.lax.axis_index(self.axis_name)
                    ]
                    for value in updated_model_values
                )
            else:
                updated_model_values = tuple(updated_model_values)

            loss = jax.lax.pmean(jnp.asarray(loss, dtype=jnp.float32), self.axis_name)
            # The current sample-mean contract uses this factor once. Do not
            # reuse it for a deliberate sparse-sum/token-mean policy.
            input_gradients = normalize_input_gradients(
                {key: input_gradients[key] for key in input_gradient_keys},
                world_size=self.world_size,
                micro_steps=1,
            )
            input_gradients = tuple(input_gradients[key] for key in input_gradient_keys)
            return (
                updated_model_values,
                padded_gradients,
                input_gradients,
                loss,
                predictions["logits"],
            )

        input_specs = tuple(P(self.axis_name) for _ in input_keys)
        input_gradient_specs = tuple(
            input_specs[input_keys.index(key)]
            for key in self.input_gradient_keys or input_keys
        )
        gradient_out_specs = tuple(self.gradient_specs)

        def reduction(gradients: Any) -> tuple[Any, ...]:
            return reduce_microstep_gradients(
                gradients,
                self.gradient_layout.leaves,
                stage=stage,
                axis_name=self.axis_name,
                world_size=self.world_size,
            )

        def transform_raw_gradients(gradients: Any) -> tuple[Any, ...]:
            return tuple(
                layout.to_padded_gradient(value).reshape(1, -1)
                for value, layout in zip(
                    gradients, self.gradient_layout.leaves, strict=True
                )
            )

        micro_kernels = build_shard_map_micro_accumulate(
            micro_impl,
            self.mesh,
            self.state_specs,
            self.state_specs[: self.model_leaf_count],
            gradient_out_specs,
            input_specs,
            input_gradient_specs,
            self.target_specs,
            reduce_gradients_fn=reduction,
            transform_raw_gradients_fn=transform_raw_gradients,
            axis_name=self.axis_name,
            retain_current_gradients=self._retain_parameter_gradients,
        )
        self.micro_fn_jit = micro_kernels.first
        self.micro_accumulate_fn_jit = micro_kernels.accumulate
        self.micro_diagnostic_first_fn_jit = micro_kernels.diagnostic_first
        self.micro_accumulate_diagnostic_fn_jit = micro_kernels.diagnostic_accumulate

        shard_update_runtime = None
        if stage in (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT):
            shard_update_runtime = NnxShardUpdatePlan.from_definition(
                self.state_definition
            ).build_runtime(
                definition=self.state_definition,
                packer=self.packer,
                state_layout=self.state_layout,
                gradient_layout=self.gradient_layout,
                gradient_treedef=self.gradient_treedef,
                world_size=self.world_size,
                axis_name=self.axis_name,
            )

        optimizer_update = _build_nnx_optimizer_update(
            stage=stage,
            packer=self.packer,
            state_layout=self.state_layout,
            gradient_layout=self.gradient_layout,
            gradient_treedef=self.gradient_treedef,
            shard_update_runtime=shard_update_runtime,
        )

        self.macro_fn_jit = build_shard_map_macro_update(
            optimizer_update,
            self.mesh,
            self.state_specs,
            gradient_out_specs,
            zero_stage=None,
            axis_name=self.axis_name,
            state_in_shardings=tuple(
                NamedSharding(self.mesh, spec) for spec in self.state_specs
            ),
            gradient_in_shardings=tuple(
                NamedSharding(self.mesh, spec) for spec in self.gradient_specs
            ),
            accumulation_steps_sharding=NamedSharding(self.mesh, P()),
            state_out_shardings=tuple(
                NamedSharding(self.mesh, spec) for spec in self.state_specs
            ),
            return_normalized_gradients=True,
            requires_axis_context=shard_update_runtime is not None,
        )

    def _materialize_call_arguments(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        if self.flat_state is None:
            raise RuntimeError(
                "distributed state is not materialized; restore the checkpoint or "
                "select fresh initialization before train_core"
            )
        keys = tuple(inputs)
        global_inputs = tuple(
            _global_array_from_local(
                inputs[key], mesh=self.mesh, axis_name=self.axis_name
            )
            for key in keys
        )
        target_values = tuple(
            _global_array_from_local(value, mesh=self.mesh, axis_name=self.axis_name)
            if spec == P("dp")
            else _replicated_array(value, mesh=self.mesh)
            for value, spec in zip(
                (targets[key] for key in self.target_keys),
                self.target_specs,
                strict=True,
            )
        )
        return global_inputs, target_values

    @staticmethod
    def _value_shape_dtype(value: Any) -> tuple[tuple[int, ...], jnp.dtype]:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or dtype is None:
            # Scalar metadata (for example a rank or world size) is converted
            # only to a tiny host value so preparation never materializes a
            # batch-sized buffer.
            scalar = jnp.asarray(value)
            shape = scalar.shape
            dtype = scalar.dtype
        return tuple(int(dimension) for dimension in shape), jnp.dtype(dtype)

    def _abstract_call_arguments(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[tuple[jax.ShapeDtypeStruct, ...], tuple[jax.ShapeDtypeStruct, ...]]:
        """Describe call buffers without materializing a second device array."""
        input_values: list[jax.ShapeDtypeStruct] = []
        for key in inputs:
            shape, dtype = self._value_shape_dtype(inputs[key])
            if not shape:
                raise ValueError(
                    f"distributed input {key!r} must have a leading batch dimension"
                )
            global_shape = (self.world_size * shape[0], *shape[1:])
            input_values.append(
                jax.ShapeDtypeStruct(
                    global_shape,
                    dtype,
                    sharding=NamedSharding(self.mesh, P(self.axis_name)),
                )
            )
        target_values: list[jax.ShapeDtypeStruct] = []
        for key, spec in zip(self.target_keys, self.target_specs, strict=True):
            shape, dtype = self._value_shape_dtype(targets[key])
            if spec == P(self.axis_name):
                if not shape:
                    raise ValueError(
                        f"distributed target {key!r} must have a leading batch dimension"
                    )
                shape = (self.world_size * shape[0], *shape[1:])
            target_values.append(
                jax.ShapeDtypeStruct(
                    shape,
                    dtype,
                    sharding=NamedSharding(self.mesh, spec),
                )
            )
        return tuple(input_values), tuple(target_values)

    def _abstract_gradient_arguments(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        return tuple(
            jax.ShapeDtypeStruct(
                (layout.padded_size,),
                jnp.dtype(layout.dtype),
                sharding=NamedSharding(self.mesh, spec),
            )
            for layout, spec in zip(
                self.gradient_layout.leaves, self.gradient_specs, strict=True
            )
        )

    def _call_signature(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        def leaf_signature(value: Any) -> tuple[Any, ...]:
            shape, dtype = self._value_shape_dtype(value)
            return shape, str(dtype)

        return (
            tuple(inputs),
            tuple(leaf_signature(inputs[key]) for key in inputs),
            tuple(leaf_signature(targets[key]) for key in self.target_keys),
        )

    def prepare_execution(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> None:
        """Lower and compile the first micro-step without executing it.

        Preparation is deliberately a side-effect-free seam: it builds the same
        input and target layout used by ``micro_step_once`` but does not advance
        the model, optimizer, counters, leases, or source work.
        """
        if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
            raise TypeError(
                "distributed preparation requires mapping inputs and targets"
            )
        target_keys = tuple(targets)
        if target_keys != self.target_keys:
            raise ValueError(
                "distributed target key order changed after JAX function construction: "
                f"expected={self.target_keys}, got={target_keys}"
            )
        keys = tuple(inputs)
        if self.flat_state is None:
            raise RuntimeError(
                "distributed state is not materialized; restore the checkpoint or "
                "select fresh initialization before train_core"
            )
        self._build_functions(keys)
        signature = self._call_signature(inputs, targets)
        if signature == getattr(self, "_prepared_execution_signature", None):
            return
        abstract_inputs, abstract_targets = self._abstract_call_arguments(
            inputs, targets
        )
        if self.micro_fn_jit is None or self.micro_accumulate_fn_jit is None:
            raise RuntimeError("distributed micro kernels were not initialized")
        first_kernel = (
            self.micro_diagnostic_first_fn_jit
            if self._retain_parameter_gradients
            and self.micro_diagnostic_first_fn_jit is not None
            else self.micro_fn_jit
        )
        accumulate_kernel = (
            self.micro_accumulate_diagnostic_fn_jit
            if self._retain_parameter_gradients
            and self.micro_accumulate_diagnostic_fn_jit is not None
            else self.micro_accumulate_fn_jit
        )
        abstract_gradients = self._abstract_gradient_arguments()
        abstract_state = self._abstract_flat_layout()
        # ``lower(...).compile()`` traces and compiles the executables without
        # invoking them, mutating training state, advancing counters, or
        # consuming source work. The first, accumulation, and macro kernels are
        # all prepared before the first real batch is acknowledged.
        first_kernel.lower(abstract_state, abstract_inputs, abstract_targets).compile()
        accumulate_kernel.lower(
            abstract_state, abstract_inputs, abstract_targets, abstract_gradients
        ).compile()
        if self.macro_fn_jit is None:
            raise RuntimeError("distributed macro kernel was not initialized")
        self.macro_fn_jit.lower(
            abstract_state,
            abstract_gradients,
            jax.ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(self.mesh, P())),
        ).compile()
        self._prepared_execution_signature = signature

    def micro_step_once(
        self, inputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> TrainStepResult:
        if not isinstance(targets, Mapping):
            raise TypeError("distributed targets must be a mapping")
        target_keys = tuple(targets)
        if target_keys != self.target_keys:
            raise ValueError(
                "distributed target key order changed after JAX function construction: "
                f"expected={self.target_keys}, got={target_keys}"
            )
        keys = tuple(inputs)
        if self.flat_state is None:
            raise RuntimeError(
                "distributed state is not materialized; restore the checkpoint or "
                "select fresh initialization before train_core"
            )
        self._build_functions(keys)
        global_inputs, target_values = self._materialize_call_arguments(inputs, targets)
        if self.micro_fn_jit is None or self.micro_accumulate_fn_jit is None:
            raise RuntimeError("distributed micro kernels were not initialized")
        raw_gradients = None
        reduced_gradients = None
        if self.accumulated_gradients is None:
            if (
                self._retain_parameter_gradients
                and self.micro_diagnostic_first_fn_jit is not None
            ):
                (
                    updated_model_values,
                    gradients,
                    raw_gradients,
                    reduced_gradients,
                    input_gradients,
                    loss,
                    logits,
                ) = self.micro_diagnostic_first_fn_jit(
                    self.flat_state, global_inputs, target_values
                )
            else:
                updated_model_values, gradients, input_gradients, loss, logits = (
                    self.micro_fn_jit(self.flat_state, global_inputs, target_values)
                )
        else:
            if (
                self._retain_parameter_gradients
                and self.micro_accumulate_diagnostic_fn_jit is not None
            ):
                (
                    updated_model_values,
                    gradients,
                    raw_gradients,
                    reduced_gradients,
                    input_gradients,
                    loss,
                    logits,
                ) = self.micro_accumulate_diagnostic_fn_jit(
                    self.flat_state,
                    global_inputs,
                    target_values,
                    self.accumulated_gradients,
                )
            else:
                (
                    updated_model_values,
                    gradients,
                    input_gradients,
                    loss,
                    logits,
                ) = self.micro_accumulate_fn_jit(
                    self.flat_state,
                    global_inputs,
                    target_values,
                    self.accumulated_gradients,
                )
        self.flat_state = (
            tuple(updated_model_values) + self.flat_state[self.model_leaf_count :]
        )
        self.accumulated_gradients = gradients
        self.micro_step += 1
        if self._retain_parameter_gradients:
            self._last_padded_gradients = (
                reduced_gradients if reduced_gradients is not None else gradients
            )
            self._last_raw_gradients = (
                raw_gradients
                if raw_gradients is not None
                else self._last_padded_gradients
            )
            self._last_reduced_gradients = (
                reduced_gradients
                if reduced_gradients is not None
                else self._last_padded_gradients
            )
            logical_result_gradients = (
                self.gradient_treedef.unflatten(
                    self.gradient_layout.from_padded_leaves(gradients)
                )
                if self.plan.zero.stage
                in (ZeroStage.REPLICATED, ZeroStage.OPTIMIZER_STATE)
                else None
            )
        else:
            logical_result_gradients = None
        return TrainStepResult(
            loss=loss,
            predictions={"logits": logits},
            gradients=logical_result_gradients,
            input_gradients={
                key: value
                for key, value in zip(
                    self.input_gradient_keys or keys, input_gradients, strict=True
                )
            },
            step=self.step,
            micro_step=self.micro_step,
            optimizer_applied=self.micro_step == self.plan.accumulation.micro_steps,
        )

    def materialize_parameter_gradients(self) -> Any:
        """Materialize the latest full parameter gradients for explicit diagnostics."""
        padded_gradients = self._last_padded_gradients
        if padded_gradients is None:
            raise RuntimeError("no distributed micro-step gradients are available")
        return self.gradient_treedef.unflatten(
            self.gradient_layout.from_padded_leaves(padded_gradients)
        )

    def apply_macro_update(self, window: OptimizerUpdateWindow) -> bool:
        if self.micro_step != self.plan.accumulation.micro_steps:
            return False
        if not window.should_apply_update:
            return False
        self._apply_accumulated_update(self.micro_step)
        return True

    def finalize_checkpoint_window(self) -> bool:
        if self.micro_step == 0:
            return False
        self._apply_accumulated_update(self.micro_step)
        return True

    def _apply_accumulated_update(self, accumulated_steps: int) -> None:
        if accumulated_steps <= 0 or self.micro_step != accumulated_steps:
            raise RuntimeError(
                "distributed accumulated gradient window is inconsistent"
            )
        if self.accumulated_gradients is None or self.macro_fn_jit is None:
            raise RuntimeError(
                "distributed accumulated gradient window is missing state"
            )
        if self.flat_state is None:
            raise RuntimeError("distributed state is not initialized")
        macro_output = self.macro_fn_jit(
            self.flat_state,
            self.accumulated_gradients,
            jnp.asarray(accumulated_steps, dtype=jnp.int32),
        )
        if not isinstance(macro_output, MacroUpdateOutput):
            raise TypeError("distributed macro kernel must return MacroUpdateOutput")
        self.flat_state = macro_output.state
        del macro_output
        self.accumulated_gradients = None
        self.micro_step = 0
        self.step += 1

    def capture_checkpoint_state(self) -> Mapping[str, Any]:
        """Capture distributed state for checkpointing with unpadded, original logical tensor shapes."""
        if self.flat_state is None:
            raise RuntimeError("distributed state is not initialized")
        if self.micro_step != 0 or self.accumulated_gradients is not None:
            raise RuntimeError(
                "distributed checkpoint capture requires a macro-update boundary"
            )
        flat_state = self._global_checkpoint_values(
            self.flat_state, self._checkpoint_value_materializers
        )
        unflattened = self.packer.unflatten_states(flat_state)
        return {
            "step": _replicated_array(
                jnp.asarray(self.step, dtype=jnp.int32), mesh=self.mesh
            ),
            "model_state": unflattened["model"],
            "optimizer_state": unflattened["optimizer"],
        }

    def checkpoint_layout(self) -> Mapping[str, Any]:
        return self.state_definition.checkpoint_layout(
            self.parallel_mesh, axis_name=self.axis_name
        )

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore checkpoint state from unpadded logical arrays, re-padding and sharding for the active mesh."""
        if self.state_lifecycle is not TrainingStateLifecycle.CHECKPOINT_RESTORE:
            raise RuntimeError(
                "restore_checkpoint_state requires the checkpoint-restore lifecycle"
            )
        if self.flat_state is not None:
            raise RuntimeError(
                "restore_checkpoint_state requires an uninitialized runtime"
            )
        step = self.state_definition.validate_checkpoint_payload(
            state, context="distributed checkpoint"
        )
        all_leaves = tuple(self.packer.iter_state_leaves(**state))
        checkpoint_sharders = self._build_checkpoint_state_sharders(
            self.state_layout, self.state_specs
        )
        if len(all_leaves) != len(checkpoint_sharders):
            raise ValueError(
                f"distributed checkpoint leaves count {len(all_leaves)} != expected {len(checkpoint_sharders)}"
            )
        for index, (leaf, layout) in enumerate(
            zip(all_leaves, self.state_layout.leaves, strict=True)
        ):
            expected_sharding = self._checkpoint_sharding(layout.original_shape)
            if not isinstance(leaf, jax.Array) or leaf.sharding != expected_sharding:
                raise ValueError(
                    "distributed checkpoint leaf must match the restore layout: "
                    f"index={index}, expected={expected_sharding}, "
                    f"actual={getattr(leaf, 'sharding', None)}"
                )
        self.flat_state = tuple(
            shard_state(leaf)
            for shard_state, leaf in zip(checkpoint_sharders, all_leaves, strict=True)
        )
        self.step = step
        self.micro_step = 0
        self.accumulated_gradients = None

    def capture_module_fingerprint(self) -> Mapping[str, Any]:
        if self.flat_state is None:
            raise RuntimeError("distributed state is not initialized")
        flat_state = self._global_checkpoint_values(
            self.flat_state[: self.model_leaf_count],
            self._checkpoint_value_materializers[: self.model_leaf_count],
        )
        model_state = self.packer.unflatten_state("model", flat_state)
        return fingerprint_state(model_state)

    def capture_audit_state(self) -> Mapping[str, Any]:
        if self.flat_state is None:
            raise RuntimeError("distributed state is not initialized")
        flat_state = self._global_checkpoint_values(
            self.flat_state, self._checkpoint_value_materializers
        )
        unflattened = self.packer.unflatten_states(flat_state)
        accumulated_fingerprint = (
            None
            if self.accumulated_gradients is None
            else fingerprint_state(self.accumulated_gradients)
        )
        raw_fingerprint = (
            None
            if self._last_raw_gradients is None
            else fingerprint_state(self._last_raw_gradients)
        )
        if raw_fingerprint is not None:
            gradient_paths = tuple(
                jtu.keystr(path)
                for path, _ in jtu.tree_flatten_with_path(
                    self.state_definition.abstract_gradient_state()
                )[0]
            )
            raw_fingerprint["non_finite_parameter_paths"] = [
                gradient_paths[int(leaf["leaf_index"])]
                for leaf in raw_fingerprint["non_finite_leaves"]
                if int(leaf["leaf_index"]) < len(gradient_paths)
            ]
        reduced_fingerprint = (
            None
            if self._last_reduced_gradients is None
            else fingerprint_state(self._last_reduced_gradients)
        )
        return {
            "module": fingerprint_state(unflattened["model"]),
            "optimizer": fingerprint_state(unflattened["optimizer"]),
            "accumulated_gradients": accumulated_fingerprint,
            "raw_gradients": raw_fingerprint,
            "reduced_gradients": reduced_fingerprint,
            "micro_step": self.micro_step,
            "grad_accum_steps": self.plan.accumulation.micro_steps,
        }


def build_distributed_nnx_executor(
    *,
    execution_plan: TrainingExecutionPlan,
    state_definition: NnxTrainingStateDefinition,
    forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
    loss_fn: Callable[[Any, Mapping[str, Any]], Any],
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int | None,
    rank0_address: str | None,
    state_lifecycle: TrainingStateLifecycle = TrainingStateLifecycle.FRESH_INITIALIZE,
    model_state: Any | None = None,
    materialize_parameter_gradients: bool = False,
    input_gradient_keys: tuple[str, ...] | None = None,
    target_shardings: Mapping[str, P] | None = None,
    validate_non_param_replication: bool = False,
) -> DistributedTrainStepExecutor:
    runtime = DistributedNnxTrainEngine(
        state_definition=state_definition,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        plan=execution_plan,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        rank0_address=rank0_address,
        state_lifecycle=state_lifecycle,
        model_state=model_state,
        materialize_parameter_gradients=materialize_parameter_gradients,
        input_gradient_keys=input_gradient_keys,
        target_shardings=target_shardings,
        validate_non_param_replication=validate_non_param_replication,
    )
    return DistributedTrainStepExecutor(
        plan=execution_plan,
        stage=execution_plan.zero.stage,
        prepare_execution=runtime.prepare_execution,
        micro_step=runtime.micro_step_once,
        micro_step_counter=lambda: runtime.micro_step,
        macro_update=runtime.apply_macro_update,
        checkpoint_finalize=runtime.finalize_checkpoint_window,
        checkpoint_capture=runtime.capture_checkpoint_state,
        checkpoint_layout=runtime.checkpoint_layout,
        checkpoint_restore=runtime.restore_checkpoint_state,
        audit_state_capture=runtime.capture_audit_state,
        module_fingerprint_capture=runtime.capture_module_fingerprint,
    )


__all__ = [
    "DistributedNnxTrainEngine",
    "build_distributed_nnx_executor",
    "initialize_distributed_jax",
]
