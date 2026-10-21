from types import SimpleNamespace

import pytest

from flare.data.pipeline.production.publish.resource_profile import (
    HardwareProfiler,
    RayClusterCapacity,
    StageTopologyPlanner,
    StreamingMemoryPlanner,
    WorkerMemoryPlanner,
)


def test_memory_probe_does_not_assume_fixed_capacity(monkeypatch):
    profiler = HardwareProfiler()
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)
    with pytest.raises(RuntimeError, match="memory capacity"):
        profiler._detect_memory()


def test_automatic_block_sizing_requires_object_store_capacity():
    with pytest.raises(ValueError, match="object-store"):
        StreamingMemoryPlanner().plan(
            object_store_memory_bytes=0,
            read_concurrency=1,
            datasource_prefetch_max_buffers=1,
            source_csv_block_size_bytes=None,
            target_min_block_size_bytes=None,
            target_max_block_size_bytes=None,
            datasource_prefetch_buffer_size_bytes=None,
        )


def test_nvidia_probe_surfaces_real_command_failure(monkeypatch):
    profiler = HardwareProfiler()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="driver error"
        ),
    )
    with pytest.raises(RuntimeError, match="GPU count probe"):
        profiler._nvidia_smi_count()


def test_worker_planning_requires_heap_capacity():
    with pytest.raises(ValueError, match="cluster_memory_bytes"):
        WorkerMemoryPlanner().plan(
            cluster_cpus=2,
            cluster_memory_bytes=0,
            read_concurrency=1,
            parse_actor_max_size=1,
            encode_actor_max_size=1,
            read_task_memory_bytes=64 * 1024**2,
            source_block_bytes=16 * 1024**2,
            target_block_bytes=16 * 1024**2,
            read_concurrency_explicit=False,
            parse_actor_max_explicit=False,
            encode_actor_max_explicit=False,
        )


def test_topology_planning_requires_object_store_capacity():
    with pytest.raises(ValueError, match="object_store_capacity_bytes"):
        StageTopologyPlanner().plan(
            backend="cpu",
            cluster_cpus=2,
            n_gpus=0,
            object_store_capacity_bytes=0,
        )


def test_gpu_actor_planning_requires_vram_capacity():
    with pytest.raises(ValueError, match="gpu_memory_bytes"):
        StageTopologyPlanner().infer_encoders_per_gpu(0)


@pytest.mark.parametrize("missing", ["CPU", "memory", "object_store_memory"])
def test_ray_cluster_capacity_requires_every_memory_domain(missing):
    resources = {
        "CPU": 2,
        "memory": 8 * 1024**3,
        "object_store_memory": 2 * 1024**3,
    }
    del resources[missing]

    with pytest.raises(RuntimeError, match=missing):
        RayClusterCapacity.from_mapping(resources)
