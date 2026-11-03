from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
for probe_path in (repo_root, str(Path(repo_root) / "src")):
    if probe_path not in sys.path:
        sys.path.insert(0, probe_path)

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from collections.abc import Callable, Mapping
from itertools import product
from tempfile import TemporaryDirectory
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.mesh import ParallelMesh
from flare.jax.training import Fp32Param, NnxTrainingStateDefinition, PrecisionPolicy
from flare.jax.training.engines.distributed import DistributedNnxTrainEngine
from flare.jax.training.optimizers import (
    create_adam_optimizer,
    create_adamw_optimizer,
    create_sgd_optimizer,
)
from flare.jax.training.state import FlatStateLayout
from flare.jax.training.strategy import (
    CheckpointShardingStrategy,
    ZeroShardingStrategy,
)
from flare.jax.training.zero import reduce_microstep_gradients
from flare.training.execution import (
    GradientAccumulationPolicy,
    MacroUpdatePolicy,
    OptimizerUpdateWindow,
    TrainingExecutionPlan,
    ZeroExecutionPolicy,
)
from flare.training.state import TrainingStateLifecycle
from flare.training.zero import ZeroStage


class CheckpointModel(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.arange(15, dtype=jnp.float32).reshape(5, 3))
        self.shardable = nnx.Param(jnp.arange(64, dtype=jnp.float32).reshape(8, 8))
        self.bias = Fp32Param(jnp.zeros((3,), dtype=jnp.float32))
        self.stat = nnx.BatchStat(jnp.zeros((3,), dtype=jnp.float32))


class CheckpointFactory:
    def create_module(self) -> CheckpointModel:
        return CheckpointModel()

    def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
        return create_adam_optimizer(module, learning_rate=0.01)


class AdamWCheckpointFactory(CheckpointFactory):
    def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
        return create_adamw_optimizer(module, learning_rate=0.01, weight_decay=0.01)


class ReductionModel(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.ones((1,), dtype=jnp.float32))
        self.stat = nnx.BatchStat(jnp.asarray(0.0, dtype=jnp.float32))


class ReductionFactory:
    def create_module(self) -> ReductionModel:
        return ReductionModel()

    def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
        return create_sgd_optimizer(module, learning_rate=0.1)


def execution_plan(
    stage: ZeroStage,
    accumulation_steps: int = 1,
    shard_scope: str = "global",
    sync_mode: str = "boundary",
) -> TrainingExecutionPlan:
    return TrainingExecutionPlan(
        zero=ZeroExecutionPolicy(stage, shard_scope=shard_scope),
        accumulation=GradientAccumulationPolicy(
            accumulation_steps, sync_mode=sync_mode
        ),
        update=MacroUpdatePolicy(),
    )


def create_runtime(
    *,
    definition: NnxTrainingStateDefinition,
    stage: ZeroStage,
    world_size: int,
    lifecycle: TrainingStateLifecycle,
    forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
    loss_fn: Callable[[Any, Mapping[str, Any]], Any],
    accumulation_steps: int = 1,
    local_zero_size: int | None = None,
    sync_mode: str = "boundary",
) -> DistributedNnxTrainEngine:
    runtime = object.__new__(DistributedNnxTrainEngine)
    runtime.plan = execution_plan(
        stage,
        accumulation_steps,
        shard_scope="node" if local_zero_size is not None else "global",
        sync_mode=sync_mode,
    )
    runtime.state_lifecycle = lifecycle
    runtime.state_definition = definition
    runtime.world_size = world_size
    runtime.local_zero_size = local_zero_size
    axes = (
        {
            "node": world_size // local_zero_size,
            runtime.plan.zero.axis_name: local_zero_size,
        }
        if local_zero_size is not None
        else {runtime.plan.zero.axis_name: world_size}
    )
    runtime.parallel_mesh = ParallelMesh.from_axes(
        devices=jax.devices()[:world_size],
        **axes,
    )
    runtime.zero_strategy = ZeroShardingStrategy.from_execution_plan(
        runtime.plan, runtime.parallel_mesh
    )
    runtime.checkpoint_strategy = CheckpointShardingStrategy(
        runtime.parallel_mesh, runtime.plan.zero.axis_name
    )
    runtime.packer = definition.packer()
    runtime.model_leaf_count = runtime.packer.leaf_count("model")
    runtime.state_layout = FlatStateLayout.from_abstract(
        definition.abstract_flat_leaves(),
        world_size if local_zero_size is None else local_zero_size,
    )
    runtime.model_state_layout = runtime.state_layout.prefix(runtime.model_leaf_count)
    param_state = definition.abstract_gradient_state()
    runtime.gradient_treedef = jtu.tree_structure(param_state)
    runtime.gradient_layout = FlatStateLayout.from_abstract(
        tuple(jtu.tree_leaves(param_state)),
        world_size if local_zero_size is None else local_zero_size,
    )
    runtime._checkpoint_value_materializers = (
        runtime._build_checkpoint_value_materializers(runtime.state_layout)
    )
    runtime.state_specs = runtime._state_specs()
    runtime.gradient_specs = tuple(
        P()
        if stage in (ZeroStage.REPLICATED, ZeroStage.OPTIMIZER_STATE)
        else P(runtime.axis_name)
        for _ in runtime.gradient_layout.leaves
    )
    runtime._runtime_state_sharders_cache = runtime._build_runtime_state_sharders(
        runtime.state_layout, runtime.state_specs
    )
    runtime.flat_state = None
    runtime.accumulated_gradients = None
    runtime._retain_parameter_gradients = False
    runtime.input_gradient_keys = None
    runtime.target_keys = ("labels", "samples_valid_count")
    runtime.target_specs = (runtime.batch_spec, P())
    runtime.micro_accumulate_fn_jit = None
    runtime.micro_accumulate_diagnostic_fn_jit = None
    runtime.forward_fn = forward_fn
    runtime.loss_fn = loss_fn
    runtime.input_keys = None
    runtime.micro_fn_jit = None
    runtime.macro_fn_jit = None
    runtime.step = 0
    runtime.micro_step = 0
    if lifecycle is TrainingStateLifecycle.FRESH_INITIALIZE:
        runtime._ensure_concrete_state()
    return runtime


def assert_fp32_floating_leaves(state: Any) -> None:
    for leaf in jtu.tree_leaves(state):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            assert dtype == jnp.float32


def prove_reduction(stage: ZeroStage) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        ReductionFactory(), precision_policy=PrecisionPolicy.bf16()
    )

    def forward(module: ReductionModel, inputs: Mapping[str, Any]) -> dict[str, Any]:
        module.stat[...] = module.stat[...] + jnp.mean(inputs["x"])
        return {"logits": inputs["x"] * module.weight.get_value()}

    runtime = create_runtime(
        definition=definition,
        stage=stage,
        world_size=2,
        lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
        forward_fn=forward,
        loss_fn=lambda predictions, targets: jnp.mean(
            (predictions["logits"] - targets["labels"]) ** 2
        ),
    )
    runtime._build_functions(("x",))
    assert runtime.flat_state is not None
    assert runtime.micro_fn_jit is not None
    state_sharding = NamedSharding(runtime.mesh, P())
    input_sharding = NamedSharding(runtime.mesh, P(runtime.axis_name))
    x = jax.make_array_from_single_device_arrays(
        (2, 1),
        input_sharding,
        [
            jax.device_put(jnp.asarray([[1.0]], dtype=jnp.float32), jax.devices()[0]),
            jax.device_put(jnp.asarray([[3.0]], dtype=jnp.float32), jax.devices()[1]),
        ],
    )
    labels = jax.make_array_from_single_device_arrays(
        (2, 1),
        input_sharding,
        [
            jax.device_put(jnp.zeros((1, 1), dtype=jnp.float32), jax.devices()[0]),
            jax.device_put(jnp.zeros((1, 1), dtype=jnp.float32), jax.devices()[1]),
        ],
    )
    valid_count = jax.device_put(jnp.asarray(2, dtype=jnp.int32), state_sharding)
    updated_model, gradients, input_gradients, loss, predictions = runtime.micro_fn_jit(
        runtime.flat_state, (x,), (labels, valid_count)
    )
    decoded_gradients = runtime.gradient_layout.from_padded_leaves(gradients)
    assert all(value.dtype == jnp.float32 for value in gradients)
    assert jnp.allclose(decoded_gradients[0], jnp.asarray([10.0], dtype=jnp.float32))
    assert runtime.micro_accumulate_fn_jit is not None
    (
        second_model,
        accumulated,
        second_input_gradients,
        second_loss,
        second_predictions,
    ) = runtime.micro_accumulate_fn_jit(
        tuple(updated_model) + runtime.flat_state[runtime.model_leaf_count :],
        (x,),
        (labels, valid_count),
        gradients,
    )
    decoded_accumulated = runtime.gradient_layout.from_padded_leaves(accumulated)
    assert jnp.allclose(decoded_accumulated[0], jnp.asarray([20.0], dtype=jnp.float32))
    assert input_gradients[0].dtype == jnp.float32
    assert second_input_gradients[0].dtype == jnp.float32
    assert loss.dtype == jnp.float32
    assert jnp.allclose(loss, 5.0)
    assert predictions.dtype == jnp.float32
    assert second_loss.dtype == jnp.float32
    assert second_predictions.dtype == jnp.float32
    runtime.flat_state = (
        tuple(second_model) + runtime.flat_state[runtime.model_leaf_count :]
    )
    runtime.accumulated_gradients = accumulated
    runtime.micro_step = 2
    runtime._apply_accumulated_update(2)
    checkpoint = runtime.capture_checkpoint_state()
    assert jnp.allclose(
        checkpoint["model_state"]["weight"].get_value(),
        0.0,
        rtol=0.0,
        atol=jnp.finfo(jnp.float32).eps,
    )
    assert jnp.allclose(checkpoint["model_state"]["stat"].get_value(), 4.0)
    assert_fp32_floating_leaves(checkpoint["model_state"])
    assert_fp32_floating_leaves(checkpoint["optimizer_state"])


def prove_cross_world_restore(stage: ZeroStage) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        CheckpointFactory(), precision_policy=PrecisionPolicy.bf16()
    )

    def forward(module: CheckpointModel, inputs: Mapping[str, Any]) -> dict[str, Any]:
        assert module.weight.get_value().dtype == jnp.bfloat16
        assert module.bias.get_value().dtype == jnp.float32
        module.stat[...] = module.stat[...] + jnp.mean(inputs["x"])
        return {
            "logits": inputs["x"] @ module.weight.get_value() + module.bias.get_value()
        }

    def loss(predictions: Any, targets: Mapping[str, Any]) -> Any:
        return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)

    source = create_runtime(
        definition=definition,
        stage=stage,
        world_size=2,
        lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
        forward_fn=forward,
        loss_fn=loss,
    )
    assert any(
        leaf.padded_size > leaf.original_size for leaf in source.state_layout.leaves
    )
    result = source.micro_step_once(
        {"x": jnp.ones((1, 5), dtype=jnp.float32)},
        {
            "labels": jnp.zeros((1, 3), dtype=jnp.float32),
            "samples_valid_count": jnp.asarray(2, dtype=jnp.int32),
        },
    )
    assert result.loss.dtype == jnp.float32
    assert result.predictions["logits"].dtype == jnp.float32
    assert result.input_gradients["x"].dtype == jnp.float32
    assert source.finalize_checkpoint_window()
    checkpoint = source.capture_checkpoint_state()
    assert set(checkpoint) == {"step", "model_state", "optimizer_state"}
    source_leaves = tuple(source.packer.iter_state_leaves(**checkpoint))
    assert all(
        leaf.shape == layout.original_shape
        for leaf, layout in zip(source_leaves, source.state_layout.leaves, strict=True)
    )
    assert checkpoint["model_state"]["shardable"].get_value().sharding.spec == P("dp")
    assert source._checkpoint_sharding((3, 8)).spec == P(None, "dp")
    assert source._checkpoint_sharding((5, 3)).spec == P()
    assert isinstance(checkpoint["model_state"]["bias"], Fp32Param)
    assert_fp32_floating_leaves(checkpoint["model_state"])
    assert_fp32_floating_leaves(checkpoint["optimizer_state"])

    restored = create_runtime(
        definition=definition,
        stage=stage,
        world_size=4,
        lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
        forward_fn=forward,
        loss_fn=loss,
    )
    assert restored.checkpoint_layout()["model_state"][
        "shardable"
    ].get_value().sharding.spec == P("dp")
    invalid_checkpoint = dict(checkpoint)
    invalid_checkpoint["step"] = jnp.asarray(1.0, dtype=jnp.float32)
    try:
        restored.restore_checkpoint_state(invalid_checkpoint)
    except TypeError:
        pass
    else:
        raise AssertionError("distributed runtime accepted a float checkpoint step")
    assert restored.flat_state is None
    try:
        restored.restore_checkpoint_state(checkpoint)
    except ValueError as error:
        assert "must match the restore layout" in str(error)
    else:
        raise AssertionError("distributed runtime accepted a different mesh")
    assert restored.flat_state is None

    with TemporaryDirectory(prefix="flare-cross-world-checkpoint-") as directory:
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        checkpoint_path = Path(directory) / "state"
        try:
            checkpointer.save(
                checkpoint_path,
                args=ocp.args.StandardSave(checkpoint),
            )
            target_checkpoint = checkpointer.restore(
                checkpoint_path,
                args=ocp.args.StandardRestore(item=restored.checkpoint_layout()),
            )
        finally:
            checkpointer.close()

    target_leaves = tuple(restored.packer.iter_state_leaves(**target_checkpoint))
    assert all(
        target.shape == layout.original_shape
        and np.array_equal(np.asarray(target), np.asarray(source_leaf))
        for target, source_leaf, layout in zip(
            target_leaves, source_leaves, restored.state_layout.leaves, strict=True
        )
    )
    assert any(
        source_layout.padded_size == 2 and target_layout.padded_size == 4
        for source_layout, target_layout in zip(
            source.state_layout.leaves, restored.state_layout.leaves, strict=True
        )
    )
    restored.restore_checkpoint_state(target_checkpoint)
    assert all(
        runtime_leaf.shape == (layout.padded_size,)
        and runtime_leaf.sharding.spec == spec
        and np.array_equal(
            np.asarray(layout.from_padded_flat(runtime_leaf)), np.asarray(target)
        )
        for runtime_leaf, layout, spec, target in zip(
            restored.flat_state,
            restored.state_layout.leaves,
            restored.state_specs,
            target_leaves,
            strict=True,
        )
    )
    restored_checkpoint = restored.capture_checkpoint_state()
    assert np.array_equal(
        np.asarray(restored_checkpoint["model_state"]["shardable"].get_value()),
        np.asarray(target_checkpoint["model_state"]["shardable"].get_value()),
    )
    try:
        restored.restore_checkpoint_state(checkpoint)
    except RuntimeError:
        pass
    else:
        raise AssertionError("distributed runtime accepted a second restore")
    restored.micro_step_once(
        {"x": jnp.ones((1, 5), dtype=jnp.float32)},
        {
            "labels": jnp.zeros((1, 3), dtype=jnp.float32),
            "samples_valid_count": jnp.asarray(4, dtype=jnp.int32),
        },
    )
    assert restored.finalize_checkpoint_window()
    continued = restored.capture_checkpoint_state()
    assert int(continued["step"]) == 2
    assert continued["model_state"]["weight"].get_value().shape == (5, 3)
    assert_fp32_floating_leaves(continued["model_state"])
    assert_fp32_floating_leaves(continued["optimizer_state"])


def prove_fused_accumulation(stage: ZeroStage) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        ReductionFactory(), precision_policy=PrecisionPolicy.bf16()
    )

    def forward(module: ReductionModel, inputs: Mapping[str, Any]) -> dict[str, Any]:
        module.stat[...] = module.stat[...] + jnp.mean(inputs["x"])
        return {"logits": inputs["x"] * module.weight.get_value()}

    runtime = create_runtime(
        definition=definition,
        stage=stage,
        world_size=2,
        lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
        forward_fn=forward,
        loss_fn=lambda predictions, targets: jnp.mean(
            (predictions["logits"] - targets["labels"]) ** 2
        ),
        accumulation_steps=3,
    )
    runtime._build_functions(("x",))
    input_sharding = NamedSharding(runtime.mesh, P("dp"))
    state_sharding = NamedSharding(runtime.mesh, P())

    def sharded(values: tuple[float, float]) -> jax.Array:
        return jax.make_array_from_single_device_arrays(
            (2, 1),
            input_sharding,
            [
                jax.device_put(
                    jnp.asarray([[values[0]]], dtype=jnp.float32), jax.devices()[0]
                ),
                jax.device_put(
                    jnp.asarray([[values[1]]], dtype=jnp.float32), jax.devices()[1]
                ),
            ],
        )

    labels = sharded((0.0, 0.0))
    valid_count = jax.device_put(jnp.asarray(2, dtype=jnp.int32), state_sharding)
    first_result = runtime.micro_step_once(
        {"x": sharded((1.0, 3.0))},
        {
            "labels": labels,
            "samples_valid_count": valid_count,
        },
    )
    first_input_gradient = first_result.input_gradients["x"]
    assert first_result.input_gradients["x"].dtype == jnp.float32
    assert len(first_result.input_gradients) == 1
    assert runtime.accumulated_gradients is not None
    assert len(runtime.accumulated_gradients) == len(runtime.gradient_layout.leaves)
    abstract_inputs, abstract_targets = runtime._abstract_call_arguments(
        {"x": sharded((1.0, 3.0))},
        {"labels": labels, "samples_valid_count": valid_count},
    )
    memory = (
        runtime.micro_accumulate_fn_jit.lower(
            runtime._abstract_flat_layout(),
            abstract_inputs,
            abstract_targets,
            runtime._abstract_gradient_arguments(),
        )
        .compile()
        .memory_analysis()
    )
    local_gradient_bytes = sum(
        leaf.addressable_shards[0].data.nbytes for leaf in runtime.accumulated_gradients
    )
    assert memory.alias_size_in_bytes >= local_gradient_bytes
    if stage is ZeroStage.OPTIMIZER_STATE:
        assert runtime.accumulated_gradients[0].sharding.spec[0] == "dp"
        assert runtime.accumulated_gradients[0].shape[0] == 2
        local_gradients = np.asarray(runtime.accumulated_gradients[0])
        assert np.allclose(local_gradients[:, 0], [2.0, 18.0])
    assert first_input_gradient is not runtime.accumulated_gradients
    assert (
        runtime.apply_macro_update(OptimizerUpdateWindow(1, 3, False, False, 0))
        is False
    )
    second_result = runtime.micro_step_once(
        {"x": sharded((2.0, 4.0))},
        {
            "labels": labels,
            "samples_valid_count": valid_count,
        },
    )
    second_input_gradient = second_result.input_gradients["x"]
    assert second_result.input_gradients["x"].dtype == jnp.float32
    assert second_input_gradient is not runtime.accumulated_gradients
    assert runtime.accumulated_gradients is not None
    macro_gradients = (
        runtime._abstract_gradient_layout()
        if runtime.macro_reduce_fn_jit is not None
        else runtime.accumulated_gradients
    )
    hlo = (
        runtime.macro_fn_jit.lower(
            runtime.flat_state,
            macro_gradients,
            jnp.asarray(2, dtype=jnp.int32),
        )
        .compiler_ir(dialect="hlo")
        .as_hlo_text()
    )
    assert "divide" in hlo
    compiled = runtime.macro_fn_jit.lower(
        runtime.flat_state,
        macro_gradients,
        jnp.asarray(2, dtype=jnp.int32),
    ).compile()
    memory = compiled.memory_analysis()
    print(
        f"MACRO_MEMORY stage={stage.value} argument={memory.argument_size_in_bytes} "
        f"output={memory.output_size_in_bytes} alias={memory.alias_size_in_bytes} "
        f"temp={memory.temp_size_in_bytes}"
    )
    gradient_bytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jtu.tree_leaves(runtime.accumulated_gradients)
    )
    if not (
        runtime._defer_gradient_sync
        and stage in (ZeroStage.GRADIENT, ZeroStage.PARAMETER)
    ):
        assert memory.alias_size_in_bytes >= gradient_bytes
    try:
        runtime.capture_checkpoint_state()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "distributed checkpoint capture must reject a partial accumulation window"
        )
    assert runtime.finalize_checkpoint_window()
    checkpoint = runtime.capture_checkpoint_state()
    assert runtime.micro_step == 0
    assert runtime.accumulated_gradients is None
    assert int(checkpoint["step"]) == 1
    assert jnp.allclose(
        checkpoint["model_state"]["weight"].get_value(),
        jnp.asarray([-0.5], dtype=jnp.float32),
        rtol=0.0,
        atol=1e-5,
    )


def prove_diagnostic_gradient_ownership() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        ReductionFactory(), precision_policy=PrecisionPolicy.bf16()
    )

    def forward(module: ReductionModel, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {"logits": inputs["x"] * module.weight.get_value()}

    runtime = create_runtime(
        definition=definition,
        stage=ZeroStage.REPLICATED,
        world_size=2,
        lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
        forward_fn=forward,
        loss_fn=lambda predictions, targets: jnp.mean(
            (predictions["logits"] - targets["labels"]) ** 2
        ),
        accumulation_steps=2,
    )
    runtime._retain_parameter_gradients = True
    for _ in range(2):
        runtime.micro_step_once(
            {"x": jnp.asarray([[1.0]], dtype=jnp.float32)},
            {
                "labels": jnp.zeros((1, 1), dtype=jnp.float32),
                "samples_valid_count": jnp.asarray(2, dtype=jnp.int32),
            },
        )
        assert runtime._last_raw_gradients is not None
        assert runtime._last_reduced_gradients is not None
        assert runtime.accumulated_gradients is not None
        raw = runtime._last_raw_gradients[0]
        assert raw.shape == (2, runtime.gradient_layout.leaves[0].padded_size)
        assert jnp.all(jnp.isfinite(raw))
        assert jnp.all(jnp.isfinite(runtime._last_reduced_gradients[0]))
        assert jnp.all(jnp.isfinite(runtime.accumulated_gradients[0]))
    print("DIAGNOSTIC_GRADIENT_OWNERSHIP=PASS")


def prove_sparse_input_gradient_dp_mean() -> None:
    from recipes.recsys.ranking.graph.losses import RankingLossAdapter

    definition = NnxTrainingStateDefinition.from_factory(
        ReductionFactory(), precision_policy=PrecisionPolicy.FP32
    )

    def base_loss(logits, labels, row_weights):
        del labels
        return jnp.sum(logits[:, 0] * row_weights)

    runtime = create_runtime(
        definition=definition,
        stage=ZeroStage.OPTIMIZER_STATE,
        world_size=2,
        lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
        forward_fn=lambda module, inputs: {
            "logits": inputs["x"] * module.weight.get_value()
        },
        loss_fn=RankingLossAdapter(base_loss),
        accumulation_steps=3,
    )
    runtime.target_keys = (
        "labels",
        "samples_valid_count",
        "global_row_start",
        "world_size",
    )
    runtime.target_specs = (P("dp"), P(), P(), P())
    runtime._build_functions(("x",))
    input_sharding = NamedSharding(runtime.mesh, P("dp"))
    scalar_sharding = NamedSharding(runtime.mesh, P())
    inputs = jax.device_put(jnp.ones((4, 1), dtype=jnp.float32), input_sharding)
    targets = {
        "labels": jax.device_put(jnp.zeros((4,), dtype=jnp.int32), input_sharding),
        "samples_valid_count": jax.device_put(
            jnp.asarray([4], dtype=jnp.int32), scalar_sharding
        ),
        "global_row_start": jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), scalar_sharding
        ),
        "world_size": jax.device_put(jnp.asarray(2, dtype=jnp.int32), scalar_sharding),
    }
    result = runtime.micro_step_once({"x": inputs}, targets)
    assert runtime.accumulated_gradients[0].sharding.spec[0] == "dp"
    reduced_gradients = tuple(
        jnp.mean(value, axis=0) for value in runtime.accumulated_gradients
    )
    dense_gradient = float(
        np.asarray(
            runtime.gradient_layout.from_padded_leaves(reduced_gradients)[0]
        ).sum()
    )
    sparse_gradient = float(np.asarray(result.input_gradients["x"]).sum())
    print(
        f"SPARSE_INPUT_GRADIENT_DP_MEAN dense={dense_gradient} "
        f"sparse={sparse_gradient} values={np.asarray(result.input_gradients['x']).tolist()}"
    )
    assert np.isclose(
        sparse_gradient,
        dense_gradient,
    ), (
        f"global-mean dense gradient {dense_gradient} != sparse gradient {sparse_gradient}"
    )
    print("SPARSE_INPUT_GRADIENT_DP_MEAN=PASS")


def prove_deferred_zero1_full_window() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        ReductionFactory(), precision_policy=PrecisionPolicy.FP32
    )
    sharded_inputs = ((1.0, 3.0), (2.0, 4.0), (3.0, 5.0))
    final_weights = {}
    for stage in (ZeroStage.REPLICATED, ZeroStage.OPTIMIZER_STATE):
        runtime = create_runtime(
            definition=definition,
            stage=stage,
            world_size=2,
            lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
            forward_fn=lambda module, inputs: {
                "logits": inputs["x"] * module.weight.get_value()
            },
            loss_fn=lambda predictions, targets: jnp.mean(
                (predictions["logits"] - targets["labels"]) ** 2
            ),
            accumulation_steps=3,
        )
        input_sharding = NamedSharding(runtime.mesh, P("dp"))
        labels = jax.device_put(jnp.zeros((2, 1), dtype=jnp.float32), input_sharding)
        for values in sharded_inputs:
            x = jax.make_array_from_single_device_arrays(
                (2, 1),
                input_sharding,
                [
                    jax.device_put(jnp.asarray([[value]], dtype=jnp.float32), device)
                    for value, device in zip(values, jax.devices()[:2], strict=True)
                ],
            )
            runtime.micro_step_once(
                {"x": x},
                {"labels": labels, "samples_valid_count": jnp.asarray(2, jnp.int32)},
            )
        assert runtime.apply_macro_update(OptimizerUpdateWindow(3, 3, True, True, 0))
        final_weights[stage] = np.asarray(
            runtime.capture_checkpoint_state()["model_state"]["weight"].get_value()
        )
    assert np.allclose(
        final_weights[ZeroStage.OPTIMIZER_STATE],
        final_weights[ZeroStage.REPLICATED],
        rtol=1e-6,
        atol=1e-6,
    )
    print("DEFERRED_ZERO1_FULL_WINDOW=PASS")


def prove_all_stages_adamw_accumulation_parity(
    policy: PrecisionPolicy = PrecisionPolicy.FP32,
) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        AdamWCheckpointFactory(), precision_policy=policy
    )
    checkpoints = {}
    for stage in ZeroStage:
        runtime = create_runtime(
            definition=definition,
            stage=stage,
            world_size=2,
            lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
            forward_fn=lambda module, inputs: {
                "logits": inputs["x"] @ module.weight.get_value()
                + module.bias.get_value()
            },
            loss_fn=lambda predictions, targets: jnp.mean(
                (predictions["logits"] - targets["labels"]) ** 2
            ),
            accumulation_steps=3,
        )
        input_sharding = NamedSharding(runtime.mesh, P("dp"))
        labels = jax.device_put(
            jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], jnp.float32),
            input_sharding,
        )
        targets = {
            "labels": labels,
            "samples_valid_count": jnp.asarray(2, jnp.int32),
        }
        for macro_index in range(2):
            for micro_index in range(3):
                base = 1.0 + macro_index * 0.2 + micro_index * 0.1
                x = jax.device_put(
                    jnp.asarray(
                        [
                            [base + i * 0.1 for i in range(5)],
                            [base + 0.6 + i * 0.2 for i in range(5)],
                        ],
                        jnp.float32,
                    ),
                    input_sharding,
                )
                runtime.micro_step_once({"x": x}, targets)
                assert all(
                    gradient.dtype == jnp.float32
                    for gradient in runtime.accumulated_gradients
                )
            if macro_index == 0:
                micro_hlo = (
                    runtime.micro_fn_jit.lower(
                        runtime._abstract_flat_layout(),
                        *runtime._abstract_call_arguments({"x": x}, targets),
                    )
                    .compiler_ir(dialect="hlo")
                    .as_hlo_text()
                )
                accumulated_micro_hlo = (
                    runtime.micro_accumulate_fn_jit.lower(
                        runtime._abstract_flat_layout(),
                        *runtime._abstract_call_arguments({"x": x}, targets),
                        runtime._abstract_gradient_arguments(),
                    )
                    .compiler_ir(dialect="hlo")
                    .as_hlo_text()
                )
                macro_gradients = (
                    runtime._abstract_gradient_layout()
                    if runtime.macro_reduce_fn_jit is not None
                    else runtime.accumulated_gradients
                )
                macro_hlo = (
                    runtime.macro_fn_jit.lower(
                        runtime.flat_state,
                        macro_gradients,
                        jnp.asarray(3, jnp.int32),
                    )
                    .compiler_ir(dialect="hlo")
                    .as_hlo_text()
                )
                reduction_hlo = (
                    runtime.macro_reduce_fn_jit.lower(
                        runtime._abstract_gradient_arguments()
                    )
                    .compiler_ir(dialect="hlo")
                    .as_hlo_text()
                    if runtime.macro_reduce_fn_jit is not None
                    else macro_hlo
                )
                hlo = micro_hlo if stage is ZeroStage.REPLICATED else reduction_hlo
                collective = "all-reduce(" if stage.value < 2 else "reduce-scatter("
                wire_dtype = "f32" if policy is PrecisionPolicy.FP32 else "bf16"
                lines = [line for line in hlo.splitlines() if collective in line]
                assert any(f"= {wire_dtype}[" in line for line in lines), hlo
                if stage is not ZeroStage.REPLICATED:
                    for micro in (micro_hlo, accumulated_micro_hlo):
                        assert not any(
                            ("all-reduce(" in line or "reduce-scatter(" in line)
                            and "= bf16[" in line
                            for line in micro.splitlines()
                        )
            assert runtime.apply_macro_update(
                OptimizerUpdateWindow(3, 3, True, True, macro_index)
            )
        checkpoint = runtime.capture_checkpoint_state()
        assert_fp32_floating_leaves(checkpoint["model_state"])
        assert_fp32_floating_leaves(checkpoint["optimizer_state"])
        checkpoints[stage] = tuple(
            np.asarray(leaf) for leaf in runtime.packer.iter_state_leaves(**checkpoint)
        )
    reference = checkpoints[ZeroStage.REPLICATED]
    for stage in (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT, ZeroStage.PARAMETER):
        assert len(checkpoints[stage]) == len(reference)
        for actual, expected in zip(checkpoints[stage], reference, strict=True):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5 if policy is PrecisionPolicy.FP32 else 2e-2,
                atol=1e-5,
            )
    print(f"ALL_STAGES_ADAMW_ACCUMULATION_PARITY precision={policy.value}=PASS")


def prove_gradient_transport() -> None:
    """Verify wire dtype, padding, averaging, and FP32 accumulation independently."""
    world_size = 4
    mesh = ParallelMesh.from_axes(devices=jax.devices(), dp=world_size).mesh
    local = jnp.asarray([1.003, 1e10, 1e-20, -1.003, -1e10, -1e-20, 0.0], jnp.float32)
    inputs = jax.device_put(
        jnp.stack([local * (rank + 1) for rank in range(world_size)]),
        NamedSharding(mesh, P("dp", None)),
    )
    layouts = FlatStateLayout.from_abstract(
        (jax.ShapeDtypeStruct(local.shape, local.dtype),), world_size
    ).leaves
    reference = np.pad(np.asarray(inputs).mean(axis=0), (0, 1))
    for policy in PrecisionPolicy:
        for stage in ZeroStage:

            def reduce(local_inputs, *, stage=stage, policy=policy):
                return reduce_microstep_gradients(
                    (local_inputs[0],),
                    layouts,
                    stage=stage,
                    axis_name="dp",
                    world_size=world_size,
                    transport_dtype=policy.gradient_transport_dtype,
                )[0]

            kernel = jax.jit(
                jax.shard_map(
                    reduce,
                    mesh=mesh,
                    in_specs=P("dp", None),
                    out_specs=P() if stage.value < 2 else P("dp"),
                    check_vma=False,
                )
            )
            result = kernel(inputs)
            assert result.dtype == jnp.float32
            assert np.isfinite(np.asarray(result)).all()
            np.testing.assert_allclose(
                np.asarray(result),
                reference,
                rtol=1e-6 if policy is PrecisionPolicy.FP32 else 1e-2,
                atol=0.0,
            )
            hlo = kernel.lower(inputs).compiler_ir(dialect="hlo").as_hlo_text()
            collective = "all-reduce(" if stage.value < 2 else "reduce-scatter("
            wire_dtype = "f32" if policy is PrecisionPolicy.FP32 else "bf16"
            lines = [line for line in hlo.splitlines() if collective in line]
            assert lines and all(f"= {wire_dtype}[" in line for line in lines), hlo
            conversions = [line for line in hlo.splitlines() if " convert(" in line]
            assert len(conversions) == (0 if policy is PrecisionPolicy.FP32 else 2), hlo
    print("GRADIENT_TRANSPORT=PASS")


def prove_node_local_zero_adamw() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        AdamWCheckpointFactory(), precision_policy=PrecisionPolicy.BF16
    )
    checkpoints = {}
    for stage, local_zero_size, sync_mode in product(
        (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT, ZeroStage.PARAMETER),
        (None, 2),
        ("boundary", "micro"),
    ):
        runtime = create_runtime(
            definition=definition,
            stage=stage,
            world_size=4,
            local_zero_size=local_zero_size,
            sync_mode=sync_mode,
            lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
            forward_fn=lambda module, inputs: {
                "logits": inputs["x"] @ module.weight.get_value()
                + module.bias.get_value()
            },
            loss_fn=lambda predictions, targets: jnp.mean(
                (predictions["logits"] - targets["labels"]) ** 2
            ),
            accumulation_steps=3,
        )
        if local_zero_size is not None:
            assert dict(runtime.mesh.shape) == {"node": 2, "dp": 2}
            assert runtime.state_specs[runtime.model_leaf_count] == P("dp")
            assert runtime.accumulation_specs[0] == (
                P(("node", "dp"), None)
                if sync_mode == "boundary"
                else P()
                if stage is ZeroStage.OPTIMIZER_STATE
                else P("dp")
            )
        batch_sharding = NamedSharding(runtime.mesh, runtime.batch_spec)
        labels = jax.device_put(jnp.zeros((4, 3), dtype=jnp.float32), batch_sharding)
        for macro_index in range(2):
            for micro_index in range(3):
                x = jax.device_put(
                    jnp.asarray(
                        [
                            [
                                0.3
                                + rank * 0.2
                                + column * 0.1
                                + macro_index * 0.03
                                + micro_index * 0.01
                                for column in range(5)
                            ]
                            for rank in range(4)
                        ],
                        dtype=jnp.float32,
                    ),
                    batch_sharding,
                )
                runtime.micro_step_once(
                    {"x": x},
                    {
                        "labels": labels,
                        "samples_valid_count": jnp.asarray(4, jnp.int32),
                    },
                )
            assert runtime.apply_macro_update(
                OptimizerUpdateWindow(3, 3, True, True, macro_index)
            )
        checkpoint = runtime.capture_checkpoint_state()
        checkpoints[(stage, local_zero_size, sync_mode)] = tuple(
            np.asarray(value)
            for value in runtime.packer.iter_state_leaves(**checkpoint)
        )
    for stage in (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT, ZeroStage.PARAMETER):
        reference = checkpoints[(stage, None, "boundary")]
        for local_zero_size, sync_mode in product((None, 2), ("boundary", "micro")):
            for expected, actual in zip(
                reference, checkpoints[(stage, local_zero_size, sync_mode)], strict=True
            ):
                np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=1e-5)
    print("NODE_LOCAL_ZERO_ADAMW=PASS")


def prove_node_local_collective_placement() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        ReductionFactory(), precision_policy=PrecisionPolicy.BF16
    )
    for stage, sync_mode in product(
        (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT, ZeroStage.PARAMETER),
        ("boundary", "micro"),
    ):
        runtime = create_runtime(
            definition=definition,
            stage=stage,
            world_size=4,
            local_zero_size=2,
            sync_mode=sync_mode,
            lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
            forward_fn=lambda module, inputs: {
                "logits": inputs["x"] * module.weight.get_value()
            },
            loss_fn=lambda predictions, targets: jnp.mean(
                (predictions["logits"] - targets["labels"]) ** 2
            ),
            accumulation_steps=3,
        )
        runtime._build_functions(("x",))
        abstract_inputs, abstract_targets = runtime._abstract_call_arguments(
            {"x": jnp.ones((1, 1), jnp.float32)},
            {
                "labels": jnp.zeros((1, 1), jnp.float32),
                "samples_valid_count": jnp.asarray(4, jnp.int32),
            },
        )
        state = runtime._abstract_flat_layout()
        gradients = runtime._abstract_gradient_arguments()
        micro_hlo = (
            runtime.micro_fn_jit.lower(state, abstract_inputs, abstract_targets)
            .compiler_ir(dialect="hlo")
            .as_hlo_text()
        )
        accumulated_micro_hlo = (
            runtime.micro_accumulate_fn_jit.lower(
                state, abstract_inputs, abstract_targets, gradients
            )
            .compiler_ir(dialect="hlo")
            .as_hlo_text()
        )
        macro_gradients = (
            runtime._abstract_gradient_layout()
            if runtime.macro_reduce_fn_jit is not None
            else gradients
        )
        macro_hlo = (
            runtime.macro_fn_jit.lower(
                state,
                macro_gradients,
                jax.ShapeDtypeStruct(
                    (), jnp.int32, sharding=NamedSharding(runtime.mesh, P())
                ),
            )
            .compiler_ir(dialect="hlo")
            .as_hlo_text()
        )
        reduction_hlo = (
            runtime.macro_reduce_fn_jit.lower(gradients)
            .compiler_ir(dialect="hlo")
            .as_hlo_text()
            if runtime.macro_reduce_fn_jit is not None
            else macro_hlo
        )

        def count(hlo: str, op: str) -> int:
            return sum(op in line and "= bf16[" in line for line in hlo.splitlines())

        if sync_mode == "boundary":
            for hlo in (micro_hlo, accumulated_micro_hlo):
                assert count(hlo, "all-reduce(") == 0
                assert count(hlo, "reduce-scatter(") == 0
            assert count(reduction_hlo, "all-reduce(") == 1
            assert count(reduction_hlo, "reduce-scatter(") == 0
        else:
            for hlo in (micro_hlo, accumulated_micro_hlo):
                assert count(hlo, "all-reduce(") == 1
                assert count(hlo, "reduce-scatter(") == 0
            assert count(macro_hlo, "all-reduce(") == 0
            assert count(macro_hlo, "reduce-scatter(") == 0
    print("NODE_LOCAL_COLLECTIVE_PLACEMENT=PASS")


def prove_node_local_checkpoint_resharding() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        AdamWCheckpointFactory(), precision_policy=PrecisionPolicy.BF16
    )
    for stage in (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT, ZeroStage.PARAMETER):
        source = create_runtime(
            definition=definition,
            stage=stage,
            world_size=4,
            lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
            forward_fn=lambda module, inputs: {
                "logits": inputs["x"] @ module.weight.get_value()
                + module.bias.get_value()
            },
            loss_fn=lambda predictions, targets: jnp.mean(
                (predictions["logits"] - targets["labels"]) ** 2
            ),
            accumulation_steps=3,
        )
        sharding = NamedSharding(source.mesh, source.batch_spec)
        x = jax.device_put(
            jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 10, sharding
        )
        labels = jax.device_put(jnp.zeros((4, 3), dtype=jnp.float32), sharding)
        for _ in range(3):
            source.micro_step_once(
                {"x": x},
                {"labels": labels, "samples_valid_count": jnp.asarray(4, jnp.int32)},
            )
        assert source.apply_macro_update(OptimizerUpdateWindow(3, 3, True, True, 0))
        original = source.capture_checkpoint_state()
        values = tuple(
            np.asarray(v) for v in source.packer.iter_state_leaves(**original)
        )
        current = original
        with TemporaryDirectory(prefix="flare-node-zero-checkpoint-") as directory:
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
            try:
                for index, local_size in enumerate((2, None)):
                    destination = create_runtime(
                        definition=definition,
                        stage=stage,
                        world_size=4,
                        local_zero_size=local_size,
                        lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
                        forward_fn=source.forward_fn,
                        loss_fn=source.loss_fn,
                        accumulation_steps=3,
                    )
                    path = Path(directory) / f"stage-{stage.value}-{index}"
                    checkpointer.save(path, args=ocp.args.StandardSave(current))
                    restored = checkpointer.restore(
                        path,
                        args=ocp.args.StandardRestore(
                            item=destination.checkpoint_layout()
                        ),
                    )
                    destination.restore_checkpoint_state(restored)
                    current = destination.capture_checkpoint_state()
                    assert int(np.asarray(current["step"])) == 1
                    for actual, expected in zip(
                        destination.packer.iter_state_leaves(**current),
                        values,
                        strict=True,
                    ):
                        np.testing.assert_allclose(
                            np.asarray(actual), expected, rtol=0, atol=0
                        )
            finally:
                checkpointer.close()
    print("NODE_LOCAL_CHECKPOINT_RESHARDING=PASS")


if __name__ == "__main__":
    for zero_stage in ZeroStage:
        prove_reduction(zero_stage)
        prove_cross_world_restore(zero_stage)
        prove_fused_accumulation(zero_stage)
        print(f"STAGE={zero_stage.value} BF16_DISTRIBUTED=PASS")

    prove_diagnostic_gradient_ownership()
    prove_sparse_input_gradient_dp_mean()
    prove_deferred_zero1_full_window()
    for precision_policy in PrecisionPolicy:
        prove_all_stages_adamw_accumulation_parity(precision_policy)
    prove_gradient_transport()
    prove_node_local_zero_adamw()
    prove_node_local_collective_placement()
    prove_node_local_checkpoint_resharding()

    print("BF16_DISTRIBUTED_RUNTIME=PASS")
