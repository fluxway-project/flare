"""Shared JAX worker environment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

JaxPlatform = Literal["cpu", "cuda"]
JaxGpuAllocator = Literal["platform", "default", "cuda_async"]


def build_jax_runtime_env(
    *,
    xla_cache_dir: str,
    platform: JaxPlatform,
    gpu_allocator: JaxGpuAllocator = "platform",
    gpu_preallocate: bool = False,
    gpu_memory_fraction: float = 0.95,
    disable_compilation_cache: bool = False,
    skip_gpu_topology_validation: bool = False,
    register_cpu_backend: bool = True,
) -> dict[str, dict[str, str]]:
    if platform not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported JAX platform: {platform!r}")
    if gpu_allocator not in {"platform", "default", "cuda_async"}:
        raise ValueError(f"unsupported JAX GPU allocator: {gpu_allocator!r}")
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0, 1]")

    env_vars = {
        "JAX_PLATFORMS": ("cuda,cpu" if register_cpu_backend else "cuda")
        if platform == "cuda"
        else "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true" if gpu_preallocate else "false",
        "JAX_COMPILATION_CACHE_DIR": xla_cache_dir,
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    }
    repository_root = Path(__file__).resolve().parents[3]
    env_vars["PYTHONPATH"] = f"{repository_root / 'src'}:{repository_root}"
    if platform == "cuda":
        env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = gpu_allocator
        if gpu_allocator != "platform":
            env_vars["XLA_CLIENT_MEM_FRACTION"] = str(gpu_memory_fraction)
        if skip_gpu_topology_validation:
            env_vars["XLA_PJRT_GPU_VALIDATE_TOPOLOGY"] = "0"
    if disable_compilation_cache:
        env_vars["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    return {"env_vars": env_vars}


__all__ = ["JaxGpuAllocator", "JaxPlatform", "build_jax_runtime_env"]
