"""Tests for PipelineConfig.create() factory method."""

from __future__ import annotations

import pytest

from flare.data.pipeline.production.publish.config import PipelineConfig


def _create_config(**overrides):
    arguments = {
        "cluster_cpus": 48,
        "cluster_memory_bytes": 512 * 1024**3,
        "object_store_capacity_bytes": 128 * 1024**3,
        "read_task_memory_bytes": 512 * 1024**2,
    }
    arguments.update(overrides)
    return PipelineConfig.create(**arguments)


class TestPipelineConfigCreate:
    """Unit tests for PipelineConfig.create() resource arithmetic."""

    def test_cpu_only_reasonable(self):
        config = _create_config(cluster_cpus=120, n_gpus=0)

        assert config.encoders_per_gpu == 1
        assert config.gpu_fraction == 0.0
        assert config.read_concurrency == 60
        assert config.write_concurrency == 30
        assert config.write_task_num_cpus == 1.0
        assert config.parse_actor_min_size == 30
        assert config.parse_actor_max_size == 30
        assert config.encode_actor_min_size == 15
        assert config.encode_actor_max_size == 15
        assert config.actor_max_tasks_in_flight == 2

    def test_gpu_auto_encoders_per_gpu(self):
        """Auto-infer encoders_per_gpu from VRAM (~6GiB each, cap 4)."""
        config = _create_config(
            cluster_cpus=120,
            n_gpus=8,
            gpu_memory_bytes=80 * 1024**3,
            gpu_mode="force",
        )

        # 80 GiB / 6 GiB = 13, capped at 4
        assert config.encoders_per_gpu == 4

    def test_gpu_mode_disable(self):
        """gpu_mode=disable should ignore available GPUs."""
        config = _create_config(
            cluster_cpus=32,
            n_gpus=4,
            gpu_memory_bytes=24 * 1024**3,
            gpu_mode="disable",
        )

        assert config.gpu_fraction == 0.0
        assert config.encoders_per_gpu == 1

    def test_read_planner_block_bytes_defaults_from_target_max_block_size(self):
        config = _create_config(cluster_cpus=120, n_gpus=0)
        assert config.read_planner_block_bytes == (config.target_max_block_size_bytes)

    def test_automatic_topology_rejects_insufficient_hardware(self):
        with pytest.raises(ValueError, match="requires at least 8 CPUs"):
            _create_config(cluster_cpus=2, n_gpus=0)

    def test_gpu_fraction_single_encoder(self):
        """Single encoder per GPU should claim full GPU."""
        config = _create_config(
            cluster_cpus=64,
            n_gpus=4,
            gpu_memory_bytes=24 * 1024**3,
            gpu_mode="force",
            encoders_per_gpu=1,
        )
        assert config.gpu_fraction == 1.0

    def test_read_memory_budget_is_per_task_not_global_concurrency_cap(self):
        config = _create_config(
            cluster_cpus=120,
            cluster_memory_bytes=8 * 1024**3,
            read_task_memory_bytes=768 * 1024**2,
            source_csv_block_size_bytes=256 * 1024 * 1024,
            datasource_prefetch_max_buffers=2,
        )
        assert config.read_concurrency == 1
        assert config.read_task_memory_bytes == 768 * 1024 * 1024
        assert config.parse_actor_max_size == 1
        assert config.encode_actor_max_size == 1

    def test_explicit_actor_pool_max_sizes(self):
        config = _create_config(
            cluster_cpus=48,
            parse_actor_max_size=11,
            encode_actor_max_size=13,
        )
        assert config.parse_actor_min_size == 11
        assert config.parse_actor_max_size == 11
        assert config.encode_actor_min_size == 13
        assert config.encode_actor_max_size == 13

    def test_explicit_stage_concurrency_overrides(self):
        config = _create_config(
            cluster_cpus=48,
            read_concurrency=7,
            write_concurrency=3,
        )
        assert config.read_concurrency == 7
        assert config.write_concurrency == 3
        assert config.read_planner_block_bytes == (config.target_max_block_size_bytes)

    def test_read_worker_heap_contract_is_explicit(self):
        with pytest.raises(ValueError, match="read_task_memory_bytes is required"):
            PipelineConfig.create(
                cluster_cpus=48,
                cluster_memory_bytes=512 * 1024**3,
                object_store_capacity_bytes=128 * 1024**3,
            )

    def test_downstream_backpressure_contract_is_explicit(self):
        config = _create_config(
            downstream_capacity_backpressure_ratio=0.5,
            dynamic_output_queue_backpressure=False,
        )

        assert config.downstream_capacity_backpressure_ratio == 0.5
        assert not config.dynamic_output_queue_backpressure
        with pytest.raises(ValueError, match="downstream_capacity_backpressure_ratio"):
            _create_config(downstream_capacity_backpressure_ratio=0.0)

    def test_streaming_generator_buffer_contract_is_explicit(self):
        assert _create_config().streaming_generator_max_blocks == 2
        assert (
            _create_config(
                streaming_generator_max_blocks=3
            ).streaming_generator_max_blocks
            == 3
        )
        with pytest.raises(ValueError, match="streaming_generator_max_blocks"):
            _create_config(streaming_generator_max_blocks=4)

    def test_eager_free_contract_is_explicit(self):
        assert _create_config(eager_free=True).eager_free
        assert not _create_config(eager_free=False).eager_free
        with pytest.raises(TypeError, match="eager_free"):
            _create_config(eager_free=1)

    def test_read_profile_contract_is_explicit(self):
        assert not _create_config().read_profile
        assert _create_config(read_profile=True).read_profile
        with pytest.raises(TypeError, match="read_profile"):
            _create_config(read_profile=1)

    def test_read_task_cpu_fraction_flows_to_topology(self):
        assert _create_config(read_task_num_cpus=0.25).read_task_num_cpus == 0.25
        with pytest.raises(ValueError, match="read_task_num_cpus"):
            _create_config(read_task_num_cpus=0.0)

    def test_stage_profile_contract_is_explicit(self):
        assert not _create_config().stage_profile
        assert _create_config(stage_profile=True).stage_profile
        with pytest.raises(TypeError, match="stage_profile"):
            _create_config(stage_profile=1)  # pyrefly: ignore[bad-argument-type]

    def test_fused_parse_encode_contract_is_explicit(self):
        assert not _create_config().fuse_parse_encode
        assert _create_config(fuse_parse_encode=True).fuse_parse_encode
        with pytest.raises(TypeError, match="fuse_parse_encode"):
            _create_config(fuse_parse_encode=1)  # pyrefly: ignore[bad-argument-type]

    def test_source_task_file_boundary_is_explicit(self):
        config = _create_config(source_files_per_task=1)

        assert config.source_files_per_task == 1
        with pytest.raises(ValueError, match="source_files_per_task"):
            _create_config(source_files_per_task=0)

    def test_explicit_writer_task_cpu_reservation(self):
        config = _create_config(
            cluster_cpus=48,
            write_concurrency=3,
            write_task_num_cpus=0.5,
        )
        assert config.write_concurrency == 3
        assert config.write_task_num_cpus == 0.5

    def test_large_cpu_node_is_bounded_by_worker_heap(self):
        config = _create_config(
            cluster_cpus=512,
            cluster_memory_bytes=256 * 1024**3,
            source_csv_block_size_bytes=32 * 1024 * 1024,
            target_max_block_size_bytes=64 * 1024 * 1024,
            target_min_block_size_bytes=32 * 1024 * 1024,
        )

        assert config.read_concurrency is not None
        assert config.read_concurrency <= 512 // 2
        assert config.parse_actor_max_size < 512
        assert config.encode_actor_max_size < 512
        assert config.parse_actor_memory_bytes > 0
        assert config.encode_actor_memory_bytes > 0

    def test_explicit_read_planner_block_bytes_is_preserved(self):
        config = _create_config(
            cluster_cpus=48,
            read_planner_block_bytes=1024 * 1024 * 1024,
        )
        assert config.read_planner_block_bytes == 1024 * 1024 * 1024

    def test_zero_overrides_raise_instead_of_fallback(self):
        with pytest.raises(ValueError):
            _create_config(cluster_cpus=48, read_planner_block_bytes=0)

    def test_actor_max_tasks_in_flight_is_explicit_and_positive(self):
        config = _create_config(
            cluster_cpus=48,
            actor_max_tasks_in_flight=3,
        )
        assert config.actor_max_tasks_in_flight == 3
        with pytest.raises(ValueError, match="actor_max_tasks_in_flight"):
            _create_config(cluster_cpus=48, actor_max_tasks_in_flight=0)

    def test_parquet_file_size_knobs_are_not_pipeline_config(self):
        import inspect

        params = inspect.signature(PipelineConfig.create).parameters
        assert "target_parquet_file_rows" not in params
        assert "target_parquet_file_bytes" not in params
        assert "target_file_size_bytes" not in params
        assert "writer_target_size_bytes" not in params
        assert "files_per_partition" not in params

        config = _create_config(cluster_cpus=48)
        assert not hasattr(config, "target_parquet_file_rows")
        assert not hasattr(config, "target_parquet_file_bytes")
        assert not hasattr(config, "target_file_size_bytes")
        assert not hasattr(config, "writer_target_size_bytes")
        assert not hasattr(config, "files_per_partition")
