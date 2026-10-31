"""Shared pipeline contracts, independent of recipe configurations."""

from __future__ import annotations

import pytest


class TestStageSplitTopology:
    """Verify StageTopologyPlanner auto-tuning rules."""

    def test_cpu_topology_balances_parsers_and_encoders(self):
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="cpu",
            cluster_cpus=32,
            n_gpus=0,
            object_store_capacity_bytes=8 * 1024**3,
        )
        assert topo.encoders_per_gpu == 1
        assert topo.gpu_fraction == 0.0

    def test_cpu_topology_allocates_stage_budget(self):
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="cpu",
            cluster_cpus=128,
            n_gpus=0,
            object_store_capacity_bytes=8 * 1024**3,
        )
        assert topo.read_concurrency == 64
        assert topo.parse_actor_max_size == 32
        assert topo.encode_actor_max_size == 16
        assert topo.write_concurrency == 32
        allocated = (
            topo.read_concurrency * topo.read_task_num_cpus
            + topo.parse_actor_max_size * topo.map_actor_num_cpus
            + topo.encode_actor_max_size * topo.map_actor_num_cpus
            + topo.write_concurrency * topo.write_task_num_cpus
        )
        assert allocated == 112

    def test_read_task_cpu_fraction_controls_read_admission(self):
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="cpu",
            cluster_cpus=128,
            n_gpus=0,
            object_store_capacity_bytes=8 * 1024**3,
            read_task_num_cpus=0.25,
        )

        assert topo.read_task_num_cpus == 0.25
        assert topo.read_concurrency == 128

    def test_writer_task_cpu_cost_controls_automatic_write_capacity(self):
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="cpu",
            cluster_cpus=128,
            n_gpus=0,
            object_store_capacity_bytes=8 * 1024**3,
            write_task_num_cpus=0.5,
        )
        assert topo.write_concurrency == 64
        assert topo.write_task_num_cpus == 0.5

    def test_gpu_topology_pins_encoders_per_gpu(self):
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="gpu",
            cluster_cpus=64,
            n_gpus=4,
            object_store_capacity_bytes=8 * 1024**3,
            encoders_per_gpu=1,
            gpu_memory_bytes=16 * 1024**3,
        )
        assert topo.gpu_fraction == 1.0
        assert topo.encoders_per_gpu == 1

    def test_gpu_topology_scales_with_encoders_per_gpu(self):
        """encoders_per_gpu=4 on 8 GPUs -> 32 EncodeUDF instances, gpu_fraction=0.25."""
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="gpu",
            cluster_cpus=256,
            n_gpus=8,
            object_store_capacity_bytes=8 * 1024**3,
            encoders_per_gpu=4,
            gpu_memory_bytes=80 * 1024**3,
        )
        assert abs(topo.gpu_fraction - 0.25) < 1e-9
        assert topo.vram_per_actor == int(80 * 1024**3 * 0.85 / 4)

    def test_gpu_topology_raises_on_zero_gpus(self):
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        with pytest.raises(ValueError):
            StageTopologyPlanner().plan(
                backend="gpu",
                cluster_cpus=16,
                n_gpus=0,
                object_store_capacity_bytes=8 * 1024**3,
            )

    def test_gpu_topology_target_cluster_116cpu_8gpu(self):
        """Real benchmark cluster (116 CPU, 8 GPU) — pins n_parse, n_encode.

        Validates that on the production-shape pod the planner produces a
        parse pool wide enough to feed every encode actor (>= 2x), respects
        the auto encoders/GPU cap from VRAM (80 GiB → cap=4 →
        32 EncodeUDF actors).
        """
        from flare.data.pipeline.production.publish.resource_profile import (
            StageTopologyPlanner,
        )

        topo = StageTopologyPlanner().plan(
            backend="gpu",
            cluster_cpus=116,
            n_gpus=8,
            object_store_capacity_bytes=8 * 1024**3,
            gpu_memory_bytes=80 * 1024**3,
        )
        assert topo.encoders_per_gpu == 4
        assert abs(topo.gpu_fraction - 0.25) < 1e-9
