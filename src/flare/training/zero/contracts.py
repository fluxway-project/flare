from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ZeroStage(IntEnum):
    REPLICATED = 0
    OPTIMIZER_STATE = 1
    GRADIENT = 2
    PARAMETER = 3


@dataclass(frozen=True, slots=True)
class ZeroPartitionContract:
    """Framework-neutral partition and collective invariants."""

    stage: ZeroStage
    axis_name: str
    parameter_partition: str
    optimizer_state_partition: str
    gradient_partition: str
    gradient_collective: str
    parameter_collective: str


def zero_partition_contract(
    stage: ZeroStage, axis_name: str = "dp"
) -> ZeroPartitionContract:
    if not isinstance(stage, ZeroStage):
        raise TypeError(f"stage must be ZeroStage, got {type(stage).__name__}")
    if not axis_name:
        raise ValueError("axis_name must be non-empty")
    replicated = "replicated"
    sharded = "sharded"
    contracts = {
        ZeroStage.REPLICATED: ZeroPartitionContract(
            stage, axis_name, replicated, replicated, replicated, "none", "none"
        ),
        ZeroStage.OPTIMIZER_STATE: ZeroPartitionContract(
            stage,
            axis_name,
            replicated,
            sharded,
            replicated,
            "all-reduce",
            "none",
        ),
        ZeroStage.GRADIENT: ZeroPartitionContract(
            stage,
            axis_name,
            replicated,
            sharded,
            sharded,
            "reduce-scatter",
            "all-gather",
        ),
        ZeroStage.PARAMETER: ZeroPartitionContract(
            stage, axis_name, sharded, sharded, sharded, "reduce-scatter", "all-gather"
        ),
    }
    return contracts[stage]


__all__ = ["ZeroPartitionContract", "ZeroStage", "zero_partition_contract"]
