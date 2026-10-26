from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointCut,
    SaveSession,
    TopologySnapshot,
)

from flare.jax.training.engines.distributed import _logical_checkpoint_leaf
from flare.jax.training.engines.replicated import ReplicatedNnxTrainEngine
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.runtime.fluxway.checkpoint.finalize import checkpoint_finalizer_registration
from flare.runtime.fluxway.workers.dynamic_embedding import DynamicSparseEmbeddingSPMD
from flare.training.execution import (
    GradientAccumulationPolicy,
    MacroUpdatePolicy,
    OptimizerUpdateWindow,
    ReplicatedTrainStepExecutor,
    TrainingExecutionPlan,
    ZeroExecutionPolicy,
)
from flare.training.state import TrainingStateLifecycle
from flare.training.zero import ZeroStage


class _Module(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.asarray([[0.2]], dtype=jnp.float32))


class _Factory:
    def create_module(self) -> _Module:
        return _Module()

    def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
        return nnx.Optimizer(module, optax.sgd(0.1), wrt=nnx.Param)


def _forward(module: _Module, inputs):
    return {"logits": inputs["x"] @ module.weight}


def _loss(predictions, targets):
    return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)


def _batch(value: float):
    return (
        {"x": jnp.asarray([[value]], dtype=jnp.float32)},
        {"labels": jnp.asarray([[0.0]], dtype=jnp.float32)},
    )


def _weight(engine: ReplicatedNnxTrainEngine) -> np.ndarray:
    assert engine.module is not None
    return np.asarray(engine.module.weight.get_value())


def test_distributed_checkpoint_capture_preserves_global_logical_leaf():
    padded = jnp.arange(8, dtype=jnp.float32)
    logical = _logical_checkpoint_leaf(
        padded,
        original_size=6,
        original_shape=(2, 3),
    )
    assert logical.shape == (2, 3)
    assert jnp.array_equal(logical, jnp.arange(6, dtype=jnp.float32).reshape(2, 3))


def test_distributed_checkpoint_capture_preserves_unpadded_leaf():
    logical = _logical_checkpoint_leaf(
        jnp.asarray([3, 5], dtype=jnp.int32),
        original_size=2,
        original_shape=(2,),
    )
    assert logical.shape == (2,)
    assert jnp.array_equal(logical, jnp.asarray([3, 5], dtype=jnp.int32))


def test_dense_checkpoint_finalization_applies_partial_window():
    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    engine = ReplicatedNnxTrainEngine(definition, _forward, _loss, grad_accum_steps=3)
    engine.initialize()
    engine.train_micro_step(*_batch(1.0))
    assert engine.finalize_checkpoint_window() is True
    assert engine.micro_step == 0
    assert engine.step == 1


def test_dense_checkpoint_capture_uses_boundary_schema():
    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    engine = ReplicatedNnxTrainEngine(definition, _forward, _loss, grad_accum_steps=1)
    engine.initialize()
    engine.train_step(*_batch(1.0))
    state = engine.capture_checkpoint_state()
    assert set(state) == {"step", "model_state", "optimizer_state"}


def test_dense_boundary_checkpoint_restores_without_transient_state():
    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    source = ReplicatedNnxTrainEngine(definition, _forward, _loss, grad_accum_steps=1)
    source.initialize()
    source.train_step(*_batch(1.0))
    state = source.capture_checkpoint_state()

    restored = ReplicatedNnxTrainEngine(
        definition,
        _forward,
        _loss,
        grad_accum_steps=1,
        state_lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
    )
    restored.restore_checkpoint_state(state)
    assert restored.step == source.step
    assert restored.micro_step == 0
    assert restored.accumulated_gradients is None
    np.testing.assert_allclose(_weight(restored), _weight(source), atol=1e-6)


def test_optimizer_update_window_uses_optimizer_step_as_window_identity():
    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    engine = ReplicatedNnxTrainEngine(definition, _forward, _loss, grad_accum_steps=1)
    engine.initialize()
    plan = TrainingExecutionPlan(
        zero=ZeroExecutionPolicy(ZeroStage.REPLICATED),
        accumulation=GradientAccumulationPolicy(1),
        update=MacroUpdatePolicy(),
    )
    executor = ReplicatedTrainStepExecutor(engine, plan)
    _, first = executor.micro_step(*_batch(1.0))
    assert first.optimizer_step == engine.step == 0
    assert executor.apply_update(first) is True
    _, second = executor.micro_step(*_batch(1.0))
    assert second.optimizer_step == engine.step == 1


def test_replicated_audit_state_exposes_gradient_contract_for_execution_trace():
    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    engine = ReplicatedNnxTrainEngine(
        definition,
        _forward,
        _loss,
        grad_accum_steps=3,
        materialize_parameter_gradients=True,
    )
    engine.initialize()
    plan = TrainingExecutionPlan(
        zero=ZeroExecutionPolicy(ZeroStage.REPLICATED),
        accumulation=GradientAccumulationPolicy(3),
        update=MacroUpdatePolicy(),
    )
    executor = ReplicatedTrainStepExecutor(engine, plan)
    executor.micro_step(*_batch(1.0))

    audit = executor.capture_audit_state()

    assert audit["raw_gradients"] is not None
    assert audit["reduced_gradients"] == audit["raw_gradients"]


def test_sparse_optimizer_window_does_not_compare_operator_step():
    owner = object.__new__(DynamicSparseEmbeddingSPMD)
    owner.gname = "demo"
    owner.grad_accum_steps = 3
    owner.ctx = SimpleNamespace(step=1)
    owner._steps_since_commit = 0
    owner._active_optimizer_step = None
    window = OptimizerUpdateWindow(
        micro_step=1,
        accumulation_steps=3,
        at_boundary=False,
        should_apply_update=False,
        optimizer_step=0,
    )
    DynamicSparseEmbeddingSPMD._resolve_optimizer_window(owner, window)
    assert owner._active_optimizer_step == 0


def test_sparse_optimizer_window_changes_only_after_commit_boundary():
    owner = object.__new__(DynamicSparseEmbeddingSPMD)
    owner.gname = "demo"
    owner.grad_accum_steps = 2
    owner._steps_since_commit = 0
    owner._active_optimizer_step = 0
    first = OptimizerUpdateWindow(1, 2, False, False, 0)
    second = OptimizerUpdateWindow(2, 2, True, True, 0)
    DynamicSparseEmbeddingSPMD._resolve_optimizer_window(owner, first)
    owner._steps_since_commit = 1
    DynamicSparseEmbeddingSPMD._resolve_optimizer_window(owner, second)
    owner._steps_since_commit = 0
    next_window = OptimizerUpdateWindow(1, 2, False, False, 1)
    DynamicSparseEmbeddingSPMD._resolve_optimizer_window(owner, next_window)
    assert owner._active_optimizer_step == 1


def test_finalizer_only_registration_has_no_storage_capability():
    events = []
    registration = checkpoint_finalizer_registration(
        "training-finalize", lambda owner, session: owner.events.append(session.cut)
    )
    assert registration.descriptor.save is None
    assert registration.descriptor.restore is None
    assert registration.save is None
    assert registration.finalize is not None
    session = SaveSession(
        instance=SimpleNamespace(events=events),
        attempt_id="attempt",
        checkpoint_id="checkpoint",
        commit_permit=CheckpointCommitPermit("attempt", 0, b"permit"),
        cut=CheckpointCut.for_step("cut", 3),
        spmd_id="training",
        rank=0,
        source_topology=TopologySnapshot(1, (), ()),
        component_id="training-finalize",
    )
    asyncio.run(registration.finalize.finalize_save(session))
    assert events == [session.cut]


def test_sparse_checkpoint_finalization_flushes_partial_window():
    # pyrefly: ignore[missing-import]
    import sparse_embedding as se

    owner = object.__new__(DynamicSparseEmbeddingSPMD)
    owner.store = se.EmbeddingStore(
        embedding_config=se.EmbeddingConfig(
            dims=[2], thread_budget=1, value_dtype="float32", key_dtype="int64"
        ),
        initializer_config=se.Uniform(min_val=0.0, max_val=0.0),
        optimizer_config=se.Sgd(lr=1.0),
        score_policy="lfu",
    )
    keys = np.asarray([7], dtype=np.int64)
    owner.store.lookup_or_insert(keys, 2)
    owner.store.accumulate_gradients(keys, np.asarray([[1.0, 1.0]], dtype=np.float32))
    owner._steps_since_commit = 1
    owner._current_step = 4
    owner._last_committed_step = None
    owner._committed_steps = 0
    owner.gname = "test"

    asyncio.run(owner.finalize_checkpoint_state(object()))
    assert owner._steps_since_commit == 0


def test_sparse_native_gradient_dtype_is_independent_of_value_dtype():
    # pyrefly: ignore[missing-import]
    import sparse_embedding as se

    keys = np.asarray([7], dtype=np.int64)
    for value_dtype, gradient_dtype in (
        ("float16", "float16"),
        ("float16", "float32"),
        ("float32", "float16"),
        ("float32", "float32"),
    ):
        config = se.EmbeddingConfig(
            dims=[2],
            thread_budget=1,
            value_dtype=value_dtype,
            gradient_dtype=gradient_dtype,
            optimizer_dtype="float32",
            key_dtype="int64",
            enable_optimizer=True,
        )
        store = se.EmbeddingStore(
            embedding_config=config,
            initializer_config=se.Uniform(min_val=0.0, max_val=0.0),
            optimizer_config=se.Sgd(lr=1.0),
            score_policy="lfu",
        )
        store.lookup_or_insert(keys, 2)

        async def accumulate(
            store=store, keys=keys, gradient_dtype=gradient_dtype
        ) -> np.ndarray:
            await store.async_accumulate_gradients(
                keys, np.asarray([[1.0, 2.0]], dtype=np.dtype(gradient_dtype))
            )
            await store.async_optimizer_step(1)
            values_capsule, _ = await store.async_lookup(keys, 2)
            return np.from_dlpack(values_capsule)

        updated_values = asyncio.run(accumulate())
        assert config.value_dtype == value_dtype
        assert config.gradient_dtype == gradient_dtype
        assert store.value_dtype() == value_dtype
        assert store.optimizer_dtype() == "float32"
        assert updated_values.dtype == np.dtype(value_dtype)


def test_sparse_embedding_default_path_skips_trace_readback_and_validation(
    monkeypatch,
) -> None:
    from fluxway.common.tensor_backend import prepare_dlpack_input
    from fluxway.common.types.tensor import TransportTensor

    from flare.runtime.fluxway.workers import dynamic_embedding
    from flare.runtime.sparse_precision import (
        prepare_sparse_native_gradient as prepare_gradient,
    )

    class _Store:
        def __init__(self) -> None:
            self.accumulated: list[tuple[np.ndarray, Any]] = []
            self.step_count = 0

        async def async_lookup_or_insert(
            self, capsule: Any, dim: int
        ) -> tuple[Any, None]:
            arr = np.zeros((2, dim), dtype=np.float32)
            return prepare_dlpack_input(arr), None

        async def async_accumulate_gradients(
            self, keys: np.ndarray, grad_data: Any
        ) -> None:
            self.accumulated.append((keys, grad_data))

        async def async_optimizer_step(self, norm_steps: int) -> None:
            self.step_count += norm_steps

    class _LookupOutput:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    real_backend = dynamic_embedding.get_backend()

    class _Backend:
        @staticmethod
        def from_dlpack(capsule):
            return real_backend.from_dlpack(capsule)

        @staticmethod
        def to_numpy(_value):
            raise AssertionError("default sparse path must not read values to host")

    def _prepare_gradient(values, **kwargs):
        assert kwargs["validate_values"] is False
        return prepare_gradient(values, **kwargs)

    monkeypatch.setattr(dynamic_embedding, "get_backend", lambda: _Backend())
    monkeypatch.setattr(
        dynamic_embedding, "prepare_sparse_native_gradient", _prepare_gradient
    )
    monkeypatch.setattr(
        dynamic_embedding,
        "require_finite_sparse_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default sparse path must not scan tensor values")
        ),
    )

    owner = object.__new__(DynamicSparseEmbeddingSPMD)
    owner.gname = "test"
    owner.rank = 0
    owner.world_size = 1
    owner.embed_dims = [4]
    owner.key_dtype = "int64"
    owner.value_dtype = "float32"
    owner.gradient_dtype = "float32"
    owner.gradient_transport_dtype = "float32"
    owner.optimizer_dtype = "float32"
    owner.grad_accum_steps = 3
    owner._optimizer_enabled = True
    owner._forward_keys_cache = {}
    owner._current_step = None
    owner._last_committed_step = None
    owner._active_optimizer_step = None
    owner._steps_since_commit = 0
    owner._committed_steps = 0
    owner.validate_tensor_values = False
    owner.LookupOutput = _LookupOutput
    owner.store = _Store()
    owner.ctx = SimpleNamespace(step=0)

    keys = np.asarray([10, 20], dtype=np.int64)
    tensor = TransportTensor(flat_values=keys)
    lookup_res = asyncio.run(owner.lookup(unique_ids_dim4=tensor))
    assert hasattr(lookup_res, "sparse_features_test_dim_4")

    grad_vals = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32
    )
    grad_tensor = TransportTensor(flat_values=grad_vals, source_step=0)
    window = OptimizerUpdateWindow(
        micro_step=1,
        accumulation_steps=3,
        at_boundary=False,
        should_apply_update=False,
        optimizer_step=0,
    )
    asyncio.run(owner.update(grads_dim4=grad_tensor, optimizer_update_window=window))
    assert owner._steps_since_commit == 1
    assert len(owner.store.accumulated) == 1
