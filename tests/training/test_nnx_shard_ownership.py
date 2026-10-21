from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import nnx

from flare.jax.training.nnx.ownership import NnxShardUpdatePlan
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.optimizers import create_adam_optimizer


class _Model(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.ones((8,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((2,), dtype=jnp.float32))
        self.step_stat = nnx.BatchStat(jnp.asarray(0.0, dtype=jnp.float32))


class _AdamFactory:
    def create_module(self) -> _Model:
        return _Model()

    def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
        return create_adam_optimizer(module, learning_rate=1e-3)


def test_shard_update_plan_maps_optimizer_state_to_canonical_param_paths() -> None:
    definition = NnxTrainingStateDefinition.from_factory(_AdamFactory())

    plan = NnxShardUpdatePlan.from_definition(definition)

    assert plan.parameter_paths == frozenset({"['weight'].value", "['bias'].value"})
    assert plan.unsupported_optimizer_paths == frozenset()


def test_generic_custom_optimizer_does_not_receive_shard_capability() -> None:
    class _UnownedFactory(_AdamFactory):
        def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
            def init_fn(params):
                return (jnp.zeros((3,), dtype=jnp.float32),)

            def update_fn(updates, state, params=None):
                return updates, state

            tx = optax.GradientTransformation(init_fn, update_fn)
            return nnx.Optimizer(module, tx, wrt=nnx.Param)

    definition = NnxTrainingStateDefinition.from_factory(_UnownedFactory())

    plan = NnxShardUpdatePlan.from_definition(definition)

    try:
        plan.validate_shardable()
    except ValueError as error:
        assert "capability" in str(error)
    else:
        raise AssertionError("generic custom optimizer must not be shardable")


def test_shard_update_plan_rejects_ambiguous_parameter_matches() -> None:
    class _AmbiguousModel(nnx.Module):
        def __init__(self) -> None:
            self.layer = nnx.Dict(weight=nnx.Param(jnp.ones((4,), dtype=jnp.float32)))
            self.weight = nnx.Param(jnp.ones((4,), dtype=jnp.float32))

    class _AmbiguousFactory:
        def create_module(self) -> _AmbiguousModel:
            return _AmbiguousModel()

        def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
            return create_adam_optimizer(module, learning_rate=1e-3)

    definition = NnxTrainingStateDefinition.from_factory(_AmbiguousFactory())
    plan = NnxShardUpdatePlan.from_definition(definition)

    assert plan.unsupported_optimizer_paths
    try:
        plan.validate_shardable()
    except ValueError as error:
        assert "unsupported paths" in str(error)
    else:
        raise AssertionError("ambiguous parameter matches must fail shardability")


def test_shard_update_plan_requires_explicit_optimizer_capability() -> None:
    class _UndeclaredFactory(_AdamFactory):
        def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
            return nnx.Optimizer(module, optax.adam(1e-3), wrt=nnx.Param)

    definition = NnxTrainingStateDefinition.from_factory(_UndeclaredFactory())
    plan = NnxShardUpdatePlan.from_definition(definition)

    try:
        plan.validate_shardable()
    except ValueError as error:
        assert "capability" in str(error)
    else:
        raise AssertionError("an undeclared optimizer must not be shardable")


def test_shard_update_plan_rejects_scalar_parameter_ownership() -> None:
    class _ScalarModel(nnx.Module):
        def __init__(self) -> None:
            self.weight = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))

    class _ScalarFactory:
        def create_module(self) -> _ScalarModel:
            return _ScalarModel()

        def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
            return create_adam_optimizer(module, learning_rate=1e-3)

    definition = NnxTrainingStateDefinition.from_factory(_ScalarFactory())
    plan = NnxShardUpdatePlan.from_definition(definition)

    try:
        plan.validate_shardable()
    except ValueError as error:
        assert "scalar Param" in str(error)
    else:
        raise AssertionError("scalar Param ownership must fail shardability")


def test_shard_update_plan_rejects_optimizer_with_grad_clip_norm() -> None:
    class _ClippedFactory(_AdamFactory):
        def create_optimizer(self, module: nnx.Module) -> nnx.Optimizer:
            return create_adam_optimizer(module, learning_rate=1e-3, grad_clip_norm=1.0)

    definition = NnxTrainingStateDefinition.from_factory(_ClippedFactory())
    assert definition.optimizer_shard_update_capability is None

    plan = NnxShardUpdatePlan.from_definition(definition)
    assert plan.optimizer_capability is None

    try:
        plan.validate_shardable()
    except ValueError as error:
        assert "capability" in str(error)
    else:
        raise AssertionError(
            "optimizer with grad_clip_norm must not possess shard capability"
        )
