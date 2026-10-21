from __future__ import annotations

import dataclasses
import ipaddress
import os
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import ray


@dataclasses.dataclass
class RayConfig:
    ray_address: str | None = None
    ray_node_ip_address: str | None = None
    ray_namespace: str | None = None


class LocalRayInitOptions(TypedDict, total=False):
    num_cpus: int
    object_store_memory: int


def initialize_ray_runtime(
    *,
    address: str | None = None,
    node_ip_address: str | None = None,
    namespace: str | None = None,
    local_init_kwargs: LocalRayInitOptions | None = None,
) -> None:
    if namespace is not None and (not namespace or namespace.strip() != namespace):
        raise ValueError("Ray namespace must be a non-empty canonical string")
    if ray.is_initialized():
        return
    resolved_address = address or os.environ.get("RAY_ADDRESS")
    local_options: LocalRayInitOptions = (
        {} if local_init_kwargs is None else local_init_kwargs
    )
    if resolved_address not in {None, "local"}:
        if node_ip_address is not None or local_options:
            raise ValueError(
                "local Ray options cannot be used with an external Ray address"
            )
        ray.init(address=resolved_address, namespace=namespace)
        return
    if node_ip_address is not None:
        try:
            ipaddress.ip_address(node_ip_address)
        except ValueError as error:
            raise ValueError(
                f"Ray node IP must be a valid IP address: {node_ip_address!r}"
            ) from error
        ray.init(
            address="local",
            namespace=namespace,
            _node_ip_address=node_ip_address,
            **local_options,
        )
        return
    ray.init(address="local", namespace=namespace, **local_options)


def build_worker_runtime_env(
    runtime_env: Mapping[str, Any],
    *,
    required_env_vars: Sequence[str] = (),
    extra_env_vars: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(runtime_env, Mapping):
        raise TypeError(
            f"runtime_env must be a mapping, got {type(runtime_env).__name__}"
        )
    configured_env_vars = runtime_env.get("env_vars", {})
    if not isinstance(configured_env_vars, Mapping):
        raise TypeError(
            f"runtime_env['env_vars'] must be a mapping, got {type(configured_env_vars).__name__}"
        )

    merged_env_vars = dict(configured_env_vars)
    for var_name in required_env_vars:
        if not isinstance(var_name, str) or not var_name.strip():
            raise ValueError("required_env_vars elements must be non-empty strings")
        val = os.environ.get(var_name)
        if not val:
            raise ValueError(f"Missing required environment variable: {var_name}")
        merged_env_vars[var_name] = val

    if extra_env_vars:
        for k, v in extra_env_vars.items():
            merged_env_vars[k] = v

    result = dict(runtime_env)
    result["env_vars"] = merged_env_vars
    return result


__all__ = [
    "LocalRayInitOptions",
    "RayConfig",
    "build_worker_runtime_env",
    "initialize_ray_runtime",
]
