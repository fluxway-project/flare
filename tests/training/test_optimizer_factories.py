from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from flare.jax.training.optimizers import (
    create_adagrad_optimizer,
    create_adam_optimizer,
    create_adamw_optimizer,
    create_default_optimizer,
    create_rmsprop_optimizer,
    create_sgd_optimizer,
)


class _Model(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.asarray([1.0], dtype=jnp.float32))


def test_adamw_factory_applies_decoupled_weight_decay() -> None:
    module = _Model()
    optimizer = create_adamw_optimizer(
        module, learning_rate=0.1, weight_decay=0.2, b1=0.0, b2=0.0, eps=1e-8
    )

    optimizer.update(module, nnx.State({"weight": nnx.Param(jnp.asarray([0.0]))}))

    assert jnp.allclose(module.weight.get_value(), jnp.asarray([0.98]), atol=1e-6)


def test_sgd_factory_supports_momentum_and_nesterov() -> None:
    module = _Model()
    optimizer = create_sgd_optimizer(
        module, learning_rate=0.1, momentum=0.9, nesterov=True
    )

    optimizer.update(module, nnx.State({"weight": nnx.Param(jnp.asarray([2.0]))}))

    assert jnp.allclose(module.weight.get_value(), jnp.asarray([0.62]), atol=1e-6)


def test_adagrad_factory_uses_optax_accumulator() -> None:
    module = _Model()
    optimizer = create_adagrad_optimizer(module, learning_rate=0.1, eps=1e-8)

    optimizer.update(module, nnx.State({"weight": nnx.Param(jnp.asarray([2.0]))}))

    expected = 1.0 - 0.1 * 2.0 / jnp.sqrt(0.1 + 2.0**2 + 1e-8)
    assert jnp.allclose(module.weight.get_value(), jnp.asarray([expected]), atol=1e-6)


def test_rmsprop_factory_uses_optax_rmsprop() -> None:
    module = _Model()
    optimizer = create_rmsprop_optimizer(module, learning_rate=0.1, decay=0.0, eps=1e-8)

    optimizer.update(module, nnx.State({"weight": nnx.Param(jnp.asarray([2.0]))}))

    assert jnp.allclose(module.weight.get_value(), jnp.asarray([0.9]), atol=1e-6)


def test_default_and_adam_factories() -> None:
    module = _Model()
    optimizer = create_default_optimizer(module, lr=0.05)
    assert optimizer is not None

    adam_opt = create_adam_optimizer(module, learning_rate=0.05, grad_clip_norm=1.0)
    assert adam_opt is not None


def test_optimizer_validations_raise() -> None:
    module = _Model()
    import pytest

    with pytest.raises(AssertionError):
        create_adam_optimizer(module, learning_rate=-0.01)

    with pytest.raises(AssertionError):
        create_adam_optimizer(module, learning_rate=float("nan"))

    with pytest.raises(ValueError):
        create_adam_optimizer(module, learning_rate=0.01, b1=1.0)
