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


def test_two_cpu_automatic_topology_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires at least 8 CPUs"):
        _create_config(
            cluster_cpus=2,
            source_csv_block_size_bytes=8 * 1024 * 1024,
            target_min_block_size_bytes=8 * 1024 * 1024,
            target_max_block_size_bytes=16 * 1024 * 1024,
        )


def test_explicit_concurrency_remains_an_extension_point() -> None:
    config = _create_config(
        cluster_cpus=8,
        read_concurrency=3,
        write_concurrency=2,
    )

    assert config.read_concurrency == 3
    assert config.write_concurrency == 2


def test_object_store_fraction_flows_to_pipeline_config() -> None:
    config = _create_config(object_store_memory_fraction=0.25)

    assert config.object_store_memory_bytes == 32 * 1024**3


def test_automatic_model_actor_pools_are_bounded_on_large_nodes() -> None:
    config = _create_config(
        cluster_cpus=64,
        cluster_memory_bytes=512 * 1024**3,
        source_csv_block_size_bytes=8 * 1024 * 1024,
        target_min_block_size_bytes=8 * 1024 * 1024,
        target_max_block_size_bytes=16 * 1024 * 1024,
    )

    assert config.parse_actor_max_size == 16
    assert config.encode_actor_max_size == 8
    assert config.write_concurrency == 16
