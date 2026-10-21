from __future__ import annotations

import os
import shlex

DISTRIBUTED_GPU_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_memory_limit_slop_factor=95",
    "--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728",
    "--xla_gpu_enable_dynamic_slice_fusion=true",
    "--xla_gpu_enable_nccl_user_buffers=true",
)


def distributed_gpu_xla_flags(existing: str = "") -> str:
    supplied = shlex.split(existing)
    preset_by_name = {flag.partition("=")[0]: flag for flag in DISTRIBUTED_GPU_FLAGS}
    seen: set[str] = set()
    result: list[str] = []
    for flag in supplied:
        name, separator, _ = flag.partition("=")
        if name in preset_by_name:
            if not separator or flag.lower() != preset_by_name[name].lower():
                raise ValueError(
                    f"XLA flag conflicts with distributed_gpu preset: {flag}"
                )
            if name in seen:
                raise ValueError(f"duplicate XLA flag: {name}")
            seen.add(name)
        result.append(flag)
    result.extend(
        flag for flag in DISTRIBUTED_GPU_FLAGS if flag.partition("=")[0] not in seen
    )
    return shlex.join(result)


if __name__ == "__main__":
    print(distributed_gpu_xla_flags(os.environ.get("XLA_FLAGS", "")))
