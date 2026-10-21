from __future__ import annotations

import optax
from flax import nnx

from flare.jax.training.optimizers.base import (
    _create_sharded_nnx_optimizer,
    _non_negative,
    _positive,
    _unit_interval,
)


def create_default_optimizer(module: nnx.Module, lr: float = 1e-3) -> nnx.Optimizer:
    return create_adam_optimizer(module, learning_rate=lr)


def create_adam_optimizer(
    module: nnx.Module,
    *,
    learning_rate: float,
    grad_clip_norm: float | None = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> nnx.Optimizer:
    _positive("learning_rate", learning_rate)
    _unit_interval("b1", b1)
    _unit_interval("b2", b2)
    _positive("eps", eps)
    return _create_sharded_nnx_optimizer(
        module,
        optax.adam(learning_rate, b1=b1, b2=b2, eps=eps),
        capability_name="adam",
        grad_clip_norm=grad_clip_norm,
    )


def create_adamw_optimizer(
    module: nnx.Module,
    *,
    learning_rate: float,
    weight_decay: float = 1e-4,
    grad_clip_norm: float | None = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    nesterov: bool = False,
) -> nnx.Optimizer:
    _positive("learning_rate", learning_rate)
    _non_negative("weight_decay", weight_decay)
    _unit_interval("b1", b1)
    _unit_interval("b2", b2)
    _positive("eps", eps)
    _non_negative("eps_root", eps_root)
    if not isinstance(nesterov, bool):
        raise TypeError("nesterov must be bool")
    return _create_sharded_nnx_optimizer(
        module,
        optax.adamw(
            learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            weight_decay=weight_decay,
            nesterov=nesterov,
        ),
        capability_name="adamw",
        grad_clip_norm=grad_clip_norm,
    )


def create_sgd_optimizer(
    module: nnx.Module,
    *,
    learning_rate: float,
    momentum: float | None = None,
    nesterov: bool = False,
    grad_clip_norm: float | None = None,
) -> nnx.Optimizer:
    _positive("learning_rate", learning_rate)
    if momentum is not None:
        _unit_interval("momentum", momentum)
        if momentum == 0 and nesterov:
            raise ValueError("nesterov requires non-zero momentum")
    if not isinstance(nesterov, bool):
        raise TypeError("nesterov must be bool")
    return _create_sharded_nnx_optimizer(
        module,
        optax.sgd(learning_rate, momentum=momentum, nesterov=nesterov),
        capability_name="sgd",
        grad_clip_norm=grad_clip_norm,
    )


def create_rmsprop_optimizer(
    module: nnx.Module,
    *,
    learning_rate: float,
    decay: float = 0.9,
    eps: float = 1e-8,
    momentum: float | None = None,
    centered: bool = False,
    grad_clip_norm: float | None = None,
) -> nnx.Optimizer:
    _positive("learning_rate", learning_rate)
    _unit_interval("decay", decay)
    _positive("eps", eps)
    if momentum is not None:
        _unit_interval("momentum", momentum)
    if not isinstance(centered, bool):
        raise TypeError("centered must be bool")
    return _create_sharded_nnx_optimizer(
        module,
        optax.rmsprop(
            learning_rate,
            decay=decay,
            eps=eps,
            momentum=momentum,
            centered=centered,
        ),
        capability_name="rmsprop",
        grad_clip_norm=grad_clip_norm,
    )


def create_adagrad_optimizer(
    module: nnx.Module,
    *,
    learning_rate: float,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
    grad_clip_norm: float | None = None,
) -> nnx.Optimizer:
    _positive("learning_rate", learning_rate)
    _non_negative("initial_accumulator_value", initial_accumulator_value)
    _positive("eps", eps)
    return _create_sharded_nnx_optimizer(
        module,
        optax.adagrad(
            learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        ),
        capability_name="adagrad",
        grad_clip_norm=grad_clip_norm,
    )


__all__ = [
    "create_adagrad_optimizer",
    "create_adam_optimizer",
    "create_adamw_optimizer",
    "create_default_optimizer",
    "create_rmsprop_optimizer",
    "create_sgd_optimizer",
]
