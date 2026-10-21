"""Pluggable parallelism and sharding strategies on top of ParallelMesh.

This module implements the Open-Closed Principle (OCP) for parallelism strategies,
decoupling physical mesh coordinates from training/inference sharding algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.mesh import ParallelMesh, state_with_sharding
from flare.training.execution.plan import TrainingExecutionPlan
from flare.training.zero.contracts import (
    ZeroPartitionContract,
    ZeroStage,
    zero_partition_contract,
)


class ShardingStrategy(Protocol):
    """Abstract protocol for parallelism and sharding strategies."""

    @property
    def mesh(self) -> ParallelMesh: ...


@dataclass(frozen=True, slots=True)
class DataParallelStrategy:
    """Strategy for standard batch-dimension data parallelism."""

    mesh: ParallelMesh
    axis_name: str = "data"

    def __post_init__(self) -> None:
        if self.axis_name not in self.mesh.axis_names:
            raise ValueError(
                f"axis {self.axis_name!r} is not present in mesh axes {self.mesh.axis_names}"
            )

    def batch_sharding(self, *extra_dims: str | None) -> NamedSharding:
        return self.mesh.named(self.axis_name, *extra_dims)

    @property
    def replicated(self) -> NamedSharding:
        return self.mesh.replicated


@dataclass(frozen=True, slots=True)
class ZeroShardingStrategy:
    """Strategy for ZeRO / FSDP parameter, optimizer state, and gradient partitioning."""

    mesh: ParallelMesh
    stage: ZeroStage
    axis_name: str = "fsdp"

    def __post_init__(self) -> None:
        if self.axis_name not in self.mesh.axis_names:
            raise ValueError(
                f"axis {self.axis_name!r} is not present in mesh axes {self.mesh.axis_names}"
            )

    @classmethod
    def from_execution_plan(
        cls,
        plan: TrainingExecutionPlan,
        mesh: ParallelMesh,
    ) -> ZeroShardingStrategy:
        if not isinstance(plan, TrainingExecutionPlan):
            raise TypeError(
                f"plan must be TrainingExecutionPlan, got {type(plan).__name__}"
            )
        return cls(
            mesh=mesh,
            stage=plan.zero.stage,
            axis_name=plan.zero.axis_name,
        )

    def zero_contract(self) -> ZeroPartitionContract:
        return zero_partition_contract(self.stage, axis_name=self.axis_name)

    def zero_parameter_sharding(self) -> NamedSharding:
        contract = self.zero_contract()
        return self._partition_sharding(contract.parameter_partition)

    def zero_optimizer_sharding(self) -> NamedSharding:
        contract = self.zero_contract()
        return self._partition_sharding(contract.optimizer_state_partition)

    def zero_gradient_sharding(self) -> NamedSharding:
        contract = self.zero_contract()
        return self._partition_sharding(contract.gradient_partition)

    def _partition_sharding(self, partition: str) -> NamedSharding:
        if partition == "replicated":
            return self.mesh.replicated
        if partition == "sharded":
            return self.mesh.named(self.axis_name)
        raise ValueError(f"unknown partition type {partition!r}")

    def zero_state_specs(
        self,
        model_leaf_count: int,
        total_leaf_count: int,
    ) -> tuple[P, ...]:
        if (
            model_leaf_count < 0
            or total_leaf_count < 0
            or model_leaf_count > total_leaf_count
        ):
            raise ValueError(
                f"invalid leaf counts: model={model_leaf_count}, total={total_leaf_count}"
            )
        contract = self.zero_contract()
        model_spec = self._partition_sharding(contract.parameter_partition).spec
        optimizer_spec = self._partition_sharding(
            contract.optimizer_state_partition
        ).spec
        return tuple(
            model_spec if i < model_leaf_count else optimizer_spec
            for i in range(total_leaf_count)
        )

    def zero_gradient_specs(self, leaf_count: int) -> tuple[P, ...]:
        if leaf_count < 0:
            raise ValueError(f"leaf_count must be non-negative, got {leaf_count}")
        spec = self.zero_gradient_sharding().spec
        return tuple(spec for _ in range(leaf_count))

    def shard_model(self, module_or_state: Any) -> Any:
        sharding = self.zero_parameter_sharding()
        if isinstance(module_or_state, nnx.Module):
            graphdef, state = nnx.split(module_or_state)
            sharded_state = state_with_sharding(state, sharding)
            return nnx.merge(graphdef, sharded_state)
        return state_with_sharding(module_or_state, sharding)

    def shard_optimizer(self, optimizer_or_state: Any) -> Any:
        sharding = self.zero_optimizer_sharding()
        if isinstance(optimizer_or_state, nnx.Optimizer):
            graphdef, state = nnx.split(optimizer_or_state)
            sharded_state = state_with_sharding(state, sharding)
            return nnx.merge(graphdef, sharded_state)
        return state_with_sharding(optimizer_or_state, sharding)


@dataclass(frozen=True, slots=True)
class CheckpointShardingStrategy:
    """Strategy for layout-aware checkpoint sharding."""

    mesh: ParallelMesh
    axis_name: str

    def __post_init__(self) -> None:
        if self.axis_name not in self.mesh.axis_names:
            raise ValueError(
                f"axis {self.axis_name!r} is not present in mesh axes {self.mesh.axis_names}"
            )

    def checkpoint_sharding(self, shape: tuple[int, ...]) -> NamedSharding:
        mesh_size = self.mesh.axis_size(self.axis_name)
        if mesh_size > 1:
            for axis, dimension in enumerate(shape):
                if dimension > 0 and dimension % mesh_size == 0:
                    prefix = (None,) * axis
                    return self.mesh.named(*prefix, self.axis_name)
        return self.mesh.replicated


__all__ = [
    "CheckpointShardingStrategy",
    "DataParallelStrategy",
    "ShardingStrategy",
    "ZeroShardingStrategy",
]
