import shlex

import pytest

from flare.jax.xla_preset import DISTRIBUTED_GPU_FLAGS, distributed_gpu_xla_flags


def test_distributed_gpu_preset_contains_existing_options() -> None:
    assert tuple(shlex.split(distributed_gpu_xla_flags())) == DISTRIBUTED_GPU_FLAGS


def test_distributed_gpu_preset_preserves_unrelated_options() -> None:
    flags = distributed_gpu_xla_flags(
        "--xla_dump_to=/tmp/hlo --xla_gpu_enable_latency_hiding_scheduler=True"
    )
    assert shlex.split(flags) == [
        "--xla_dump_to=/tmp/hlo",
        "--xla_gpu_enable_latency_hiding_scheduler=True",
        *DISTRIBUTED_GPU_FLAGS[1:],
    ]


@pytest.mark.parametrize(
    "existing",
    (
        "--xla_gpu_enable_latency_hiding_scheduler=false",
        "--xla_gpu_enable_latency_hiding_scheduler",
        "--xla_gpu_memory_limit_slop_factor=80",
        "--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_memory_limit_slop_factor=95",
    ),
)
def test_distributed_gpu_preset_rejects_conflicts(existing: str) -> None:
    with pytest.raises(ValueError):
        distributed_gpu_xla_flags(existing)
