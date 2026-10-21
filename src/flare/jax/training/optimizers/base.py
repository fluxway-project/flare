"""Base primitives and capability tokens for distributed NNX optimizers.

Mathematical Principles & Design Motivation:
--------------------------------------------
In ZeRO-1 and ZeRO-2 distributed training, optimizer states (such as momentum and
variance accumulators in Adam/AdamW) and parameter updates are partitioned across
distributed devices to reduce per-device accelerator memory footprint.

Rather than maintaining a parallel optimizer codebase or rewriting Optax
transformation internals, Flare leverages the canonical Flax NNX optimizer step
by slicing parameters and gradients into local shard leaves before the update,
running the standard NNX optimizer update on each shard, and gathering updated
parameters where replication is required.

This sharded execution is mathematically exact if and only if the optimizer update
operator commutes with the slicing operator:
    S_r(Update(W, G; theta)) = Update(S_r(W), S_r(G); theta)
where S_r is the rank-r slicing operator along the flattened parameter dimension.

Commutativity holds for coordinate-wise (elementwise) updates (SGD, Adam, AdamW,
RMSprop, Adagrad) with scalar hyperparameters, because each coordinate update
w_i <- Update(w_i, g_i) depends strictly on its own coordinate's historical
moments and current gradient.

However, commutativity breaks down when cross-coordinate reductions are present:
1. Global gradient norm clipping (optax.clip_by_global_norm): Computing
   sqrt(sum(g_i^2)) locally on a shard yields a local norm that is strictly less
   than or equal to the global norm, causing each rank to scale its local gradients
   by a divergent, mathematically invalid factor.
2. Layer-wise adaptive clipping (LARS / LAMB): Calculating trust ratios ||w||_2 / ||g||_2
   on local slices produces different scaling across shards of the same tensor.

Why Explicit Capability Tokens (ShardUpdateCapability)?
-------------------------------------------------------
Inferring shardability heuristically by inspecting Optax state leaf names (e.g.,
searching for "mu" or "nu") or checking tensor shapes is fragile and dangerous.
Because Optax transformations are arbitrarily composable via optax.chain, a user
can easily chain optax.clip_by_global_norm with AdamW. A heuristic shape checker
would observe identical pytree leaf shapes and silently perform invalid local
norm clipping, causing subtle numerical corruption during training.

To prevent silent numerical bugs, Flare requires an explicit proof token
(ShardUpdateCapability). Only trusted optimizer factories that have verified that
the transformation is strictly elementwise and free from local global-norm reductions
may attach this capability.

Why Subclass ShardedNnxOptimizer(nnx.Optimizer)?
------------------------------------------------
Flax NNX uses an explicit object graph and type system (nnx.State, nnx.GraphDef,
nnx.split, nnx.eval_shape). Dynamically assigning arbitrary attributes (e.g.
optimizer._flare_shard_update_capability = ...) on a vanilla nnx.Optimizer:
1. Violates static type checkers (Pyrefly / mypy) with missing-attribute errors.
2. Risks attribute loss or untyped propagation across NNX graph transformations.
3. Conforms to the Open-Closed Principle and Liskov Substitution Principle (LSP):
   ShardedNnxOptimizer is a fully compliant nnx.Optimizer that explicitly declares
   its distributed execution capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import optax
from flax import nnx


@dataclass(frozen=True, slots=True)
class ShardUpdateCapability:
    """Proof token asserting that an optimizer's update rule commutes with slicing.

    A distributed sharded optimizer (ZeRO-1 / ZeRO-2) partitions parameter leaves
    and optimizer state leaves across devices. Slicing a parameter leaf W and its
    gradient G into a local shard S_r(W), S_r(G) before executing the optimizer
    update is mathematically valid if and only if:
        S_r(Update(W, G)) == Update(S_r(W), S_r(G))

    This holds strictly for elementwise updates with scalar hyperparameters (such
    as Adam, AdamW, SGD, RMSprop, and Adagrad). It fails when cross-coordinate
    reductions exist (e.g., global gradient norm clipping inside Optax).

    The token is intentionally explicit and opt-in:
    - It must be issued by the optimizer factory that constructed the transformation.
    - It is never guessed or heuristically inferred from Optax state leaf names.
    - If global gradient clipping is requested via `grad_clip_norm`, the capability
      is omitted because local norm clipping on shards is mathematically invalid.
    """

    name: str
    ownership_contract: str = "per_param_elementwise"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sharded optimizer capability name must be non-empty")
        if self.ownership_contract != "per_param_elementwise":
            raise ValueError(
                "sharded optimizer updates require the per_param_elementwise "
                f"ownership contract, got {self.ownership_contract!r}"
            )


def _positive(name: str, value: float) -> float:
    chex.assert_tree_all_finite(value)
    chex.assert_scalar_positive(value)
    return value


def _non_negative(name: str, value: float) -> float:
    chex.assert_tree_all_finite(value)
    chex.assert_scalar_non_negative(value)
    return value


def _unit_interval(name: str, value: float) -> float:
    chex.assert_tree_all_finite(value)
    chex.assert_scalar_in(value, 0.0, 1.0, included=True)
    if value >= 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")
    return value


class ShardedNnxOptimizer(nnx.Optimizer):
    """Flax NNX Optimizer subclass carrying explicit sharded update capability metadata.

    Subclassing `nnx.Optimizer` rather than monkey-patching dynamic attributes:
    1. Satisfies static type checkers (Pyrefly, mypy) by statically declaring the
       `_flare_shard_update_capability` attribute.
    2. Adheres to the Open-Closed and Liskov Substitution principles: `ShardedNnxOptimizer`
       is an `nnx.Optimizer` everywhere in Flax NNX pipelines (supporting `nnx.split`,
       `nnx.merge`, `nnx.eval_shape`, and checkpointing) while providing compile-time
       contracts for Flare's ZeRO distributed engines.
    3. Adds zero runtime execution overhead during JIT tracing.
    """

    _flare_shard_update_capability: ShardUpdateCapability | None

    def __init__(
        self,
        module: nnx.Module,
        tx: optax.GradientTransformation,
        *,
        capability: ShardUpdateCapability | None = None,
        wrt: Any = nnx.Param,
    ) -> None:
        super().__init__(module, tx, wrt=wrt)
        self._flare_shard_update_capability = capability


def create_nnx_optimizer(
    module: nnx.Module,
    transformation: optax.GradientTransformation,
    *,
    grad_clip_norm: float | None = None,
) -> nnx.Optimizer:
    """Creates a standard Flax NNX optimizer without sharded update capability.

    If `grad_clip_norm` is provided, `optax.clip_by_global_norm` is prepended to the
    transformation chain. The returned optimizer is a standard `nnx.Optimizer` that
    is executed over full, non-partitioned parameter states.
    """
    transformations = []
    if grad_clip_norm is not None:
        transformations.append(
            optax.clip_by_global_norm(_positive("grad_clip_norm", grad_clip_norm))
        )
    transformations.append(transformation)
    tx = optax.chain(*transformations) if len(transformations) > 1 else transformation
    return nnx.Optimizer(module, tx, wrt=nnx.Param)


def _create_sharded_nnx_optimizer(
    module: nnx.Module,
    transformation: optax.GradientTransformation,
    *,
    capability_name: str,
    grad_clip_norm: float | None,
) -> nnx.Optimizer:
    """Creates an NNX optimizer, attaching ShardUpdateCapability only if mathematically sound.

    If `grad_clip_norm` is specified, global gradient clipping is inserted into the
    transformation chain. Because `clip_by_global_norm` cannot be computed on local
    shards without cross-device communication, the resulting optimizer is NOT
    shard-update capable. In this case, a standard `nnx.Optimizer` is returned without
    capability tokens. Any subsequent attempt to execute sharded updates with this
    optimizer will fail fast during pre-trace validation (`NnxShardUpdatePlan.validate_shardable`).

    When `grad_clip_norm` is None and the transformation is elementwise, a `ShardedNnxOptimizer`
    is returned with an explicit `ShardUpdateCapability(name=capability_name)`.
    """
    if grad_clip_norm is not None:
        return create_nnx_optimizer(
            module, transformation, grad_clip_norm=grad_clip_norm
        )
    capability = ShardUpdateCapability(
        name=capability_name,
        ownership_contract="per_param_elementwise",
    )
    return ShardedNnxOptimizer(
        module, transformation, capability=capability, wrt=nnx.Param
    )


__all__ = [
    "ShardUpdateCapability",
    "ShardedNnxOptimizer",
    "_create_sharded_nnx_optimizer",
    "_non_negative",
    "_positive",
    "_unit_interval",
    "create_nnx_optimizer",
]
