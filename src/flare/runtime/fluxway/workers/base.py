from __future__ import annotations

import math
from typing import Any

from fluxway.common.types.tensor import TransportTensor


def require_operator_step(owner: Any, op_name: str) -> int:
    ctx = getattr(owner, "ctx", None)
    step = getattr(ctx, "step", None)
    if step is None:
        raise ValueError(f"[{op_name}] ctx.step is required for step-token execution.")
    return int(step)


def assigned_worker_cpu_count(runtime_context: Any, role_name: str = "Worker") -> int:
    resources = runtime_context.get_assigned_resources()
    if "CPU" not in resources:
        raise RuntimeError(f"{role_name} has no assigned CPU resource")
    cpu_value = float(resources["CPU"])
    if not math.isfinite(cpu_value) or cpu_value < 1 or not cpu_value.is_integer():
        raise ValueError(
            f"{role_name} CPU allocation must be a positive integer, "
            f"got {resources['CPU']!r}"
        )
    return int(cpu_value)


def extract_tensor_payload(value: Any) -> Any:
    if isinstance(value, TransportTensor):
        return value.flat_values
    if hasattr(value, "flat_values"):
        return value.flat_values
    return value


def extract_tensor_source_step(value: Any) -> int | None:
    if isinstance(value, TransportTensor):
        return value.source_step
    return getattr(value, "source_step", None)


__all__ = [
    "assigned_worker_cpu_count",
    "extract_tensor_payload",
    "extract_tensor_source_step",
    "require_operator_step",
]
