from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from flare.jax.layers.base import MLP
from flare.jax.layers.initialization import require_rngs
from flare.jax.layers.numerics import fp32_softmax, fp32_sum


class PLE(nnx.Module):
    """Progressive Layered Extraction. base_layers: [N+1], output: [N+1]."""

    def __init__(
        self,
        base_dim: int,
        gate_in_dim: int,
        num_experts: int,
        dims: list[int],
        n_tasks: int,
        activate: str = "relu",
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        self.num_experts = num_experts
        self.out_dim = dims[-1]
        self.n_tasks = n_tasks
        self.experts = nnx.List()
        for _ in range(n_tasks):
            for _ in range(num_experts):
                self.experts.append(MLP(base_dim, dims, activate=activate, rngs=rngs))
        i_shared = n_tasks - 1
        self.gate_layers = nnx.List()
        for i in range(n_tasks):
            gate_dims = num_experts * 2 if i != i_shared else num_experts * n_tasks
            self.gate_layers.append(nnx.Linear(gate_in_dim, gate_dims, rngs=rngs))

    def __call__(
        self, base_layers: list[jnp.ndarray], gate_layer: jnp.ndarray
    ) -> list[jnp.ndarray]:
        n_tasks = len(base_layers)
        expert_outputs = []
        idx = 0
        for i in range(n_tasks):
            local = []
            for _ in range(self.num_experts):
                local.append(self.experts[idx](base_layers[i]))
                idx += 1
            expert_outputs.append(local)
        i_shared = n_tasks - 1
        output_layers = []
        for i in range(n_tasks):
            layers = (
                expert_outputs[i] + expert_outputs[i_shared]
                if i != i_shared
                else [e for row in expert_outputs for e in row]
            )
            concat_layer = jnp.stack(layers, axis=-1)
            gate = jnp.expand_dims(
                fp32_softmax(self.gate_layers[i](gate_layer)), axis=1
            )
            output_layers.append(fp32_sum(concat_layer * gate, axis=-1))
        return output_layers


class CGC(nnx.Module):
    """Customized Gate Control. base_layers: [N+1], output: [N]."""

    def __init__(
        self,
        base_dim: int,
        gate_in_dim: int,
        num_experts: int,
        dims: list[int],
        n_tasks: int,
        activate: str = "relu",
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        self.num_experts = num_experts
        self.out_dim = dims[-1]
        self.n_tasks = n_tasks
        self.experts = nnx.List()
        for _ in range(n_tasks):
            for _ in range(num_experts):
                self.experts.append(MLP(base_dim, dims, activate=activate, rngs=rngs))
        self.gate_layers = nnx.List()
        for _ in range(n_tasks - 1):
            self.gate_layers.append(nnx.Linear(gate_in_dim, num_experts * 2, rngs=rngs))

    def __call__(
        self, base_layers: list[jnp.ndarray], gate_layer: jnp.ndarray
    ) -> list[jnp.ndarray]:
        n_tasks = len(base_layers)
        expert_outputs = []
        idx = 0
        for i in range(n_tasks):
            local = []
            for _ in range(self.num_experts):
                local.append(self.experts[idx](base_layers[i]))
                idx += 1
            expert_outputs.append(local)
        i_shared = n_tasks - 1
        output_layers = []
        for i in range(n_tasks - 1):
            layers = expert_outputs[i] + expert_outputs[i_shared]
            concat_layer = jnp.stack(layers, axis=-1)
            gate = jnp.expand_dims(
                fp32_softmax(self.gate_layers[i](gate_layer)), axis=1
            )
            output_layers.append(fp32_sum(concat_layer * gate, axis=-1))
        return output_layers


class UPLE(nnx.Module):
    """Progressive Layered Extraction with per-task gates. base_layers: [N+1], gate_layers_in: [N+1], output: [N+1]."""

    def __init__(
        self,
        base_dim: int,
        gate_in_dim: int,
        num_experts: int,
        dims: list[int],
        n_tasks: int,
        activate: str = "relu",
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        self.num_experts = num_experts
        self.out_dim = dims[-1]
        self.n_tasks = n_tasks
        self.experts = nnx.List()
        for _ in range(n_tasks):
            for _ in range(num_experts):
                self.experts.append(MLP(base_dim, dims, activate=activate, rngs=rngs))
        i_shared = n_tasks - 1
        self.gate_layers = nnx.List()
        for i in range(n_tasks):
            gate_dims = num_experts * 2 if i != i_shared else num_experts * n_tasks
            self.gate_layers.append(nnx.Linear(gate_in_dim, gate_dims, rngs=rngs))

    def __call__(
        self, base_layers: list[jnp.ndarray], gate_layers_in: list[jnp.ndarray]
    ) -> list[jnp.ndarray]:
        n_tasks = len(base_layers)
        expert_outputs = []
        idx = 0
        for i in range(n_tasks):
            local = []
            for _ in range(self.num_experts):
                local.append(self.experts[idx](base_layers[i]))
                idx += 1
            expert_outputs.append(local)
        i_shared = n_tasks - 1
        output_layers = []
        for i in range(n_tasks):
            layers = (
                expert_outputs[i] + expert_outputs[i_shared]
                if i != i_shared
                else [e for row in expert_outputs for e in row]
            )
            concat_layer = jnp.stack(layers, axis=-1)
            gate = jnp.expand_dims(
                fp32_softmax(self.gate_layers[i](gate_layers_in[i])), axis=1
            )
            output_layers.append(fp32_sum(concat_layer * gate, axis=-1))
        return output_layers


class UCGC(nnx.Module):
    """Customized Gate Control with per-task gates. base_layers: [N+1], gate_layers_in: [N], output: [N]."""

    def __init__(
        self,
        base_dim: int,
        gate_in_dim: int,
        num_experts: int,
        dims: list[int],
        n_tasks: int,
        activate: str = "relu",
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        self.num_experts = num_experts
        self.out_dim = dims[-1]
        self.n_tasks = n_tasks
        self.experts = nnx.List()
        for _ in range(n_tasks):
            for _ in range(num_experts):
                self.experts.append(MLP(base_dim, dims, activate=activate, rngs=rngs))
        self.gate_layers = nnx.List()
        for _ in range(n_tasks - 1):
            self.gate_layers.append(nnx.Linear(gate_in_dim, num_experts * 2, rngs=rngs))

    def __call__(
        self, base_layers: list[jnp.ndarray], gate_layers_in: list[jnp.ndarray]
    ) -> list[jnp.ndarray]:
        n_tasks = len(base_layers)
        expert_outputs = []
        idx = 0
        for i in range(n_tasks):
            local = []
            for _ in range(self.num_experts):
                local.append(self.experts[idx](base_layers[i]))
                idx += 1
            expert_outputs.append(local)
        i_shared = n_tasks - 1
        output_layers = []
        for i in range(n_tasks - 1):
            layers = expert_outputs[i] + expert_outputs[i_shared]
            concat_layer = jnp.stack(layers, axis=-1)
            gate = jnp.expand_dims(
                fp32_softmax(self.gate_layers[i](gate_layers_in[i])), axis=1
            )
            output_layers.append(fp32_sum(concat_layer * gate, axis=-1))
        return output_layers


__all__ = ["CGC", "PLE", "UCGC", "UPLE"]
