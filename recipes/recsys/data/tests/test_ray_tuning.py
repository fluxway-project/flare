import pyarrow as pa
import pytest

from flare.data.pipeline.production.publish.resource_profile import (
    WorkerMemoryCapacityError,
)
from recipes.recsys.data.ray_tuning import (
    MIB,
    build_cpu_iceberg_config,
    resolve_auto_tune,
)


def _schema(row_width: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                "features",
                pa.list_(pa.int64()),
                metadata={b"pad_length": str(row_width // 8).encode()},
            )
        ]
    )


def _options(**overrides):
    return {
        "cluster_cpus": 32,
        "cluster_memory_bytes": 64 * 1024**3,
        "object_store_capacity_bytes": 16 * 1024**3,
        "gpu_mode": "disable",
        **overrides,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"output_format": "parquet"}, False),
        ({"execution_engine": "heap-stream"}, False),
        ({"gpu_mode": "force"}, False),
        ({"gpu_count": 8}, False),
        ({"read_profile": True}, False),
        ({"stage_profile": True}, False),
        ({"gpu_mode": "disable", "gpu_count": 8}, True),
        ({"requested": False}, False),
        ({"requested": True, "gpu_count": 8}, False),
    ],
)
def test_auto_tune_default_is_limited_to_cpu_iceberg_and_can_be_overridden(
    overrides, expected
):
    values = {
        "requested": True,
        "execution_engine": "ray-data",
        "output_format": "iceberg",
        "gpu_mode": "auto",
        "gpu_count": 0,
        "read_profile": False,
        "stage_profile": False,
    }
    values.update(overrides)
    assert resolve_auto_tune(**values) is expected


def test_cpu_iceberg_profile_uses_schema_width_and_fuses_stages():
    narrow = build_cpu_iceberg_config(options=_options(), output_schema=_schema(18_332))
    wide = build_cpu_iceberg_config(options=_options(), output_schema=_schema(50_896))

    assert narrow.profile == "recipe-cpu-auto-v1"
    assert narrow.estimated_row_bytes == 18_336
    assert narrow.config.fuse_parse_encode
    assert narrow.config.encoded_batch_target_bytes == 128 * MIB
    assert narrow.config.read_task_num_cpus == 0.25
    assert narrow.config.write_task_num_cpus == 0.25
    assert wide.config.encoded_batch_target_bytes == 256 * MIB
    assert wide.config.target_max_block_size_bytes == 512 * MIB


def test_explicit_resource_and_batch_options_override_profile():
    tuned = build_cpu_iceberg_config(
        options=_options(
            read_concurrency=3,
            read_task_num_cpus=1,
            write_concurrency=2,
            write_task_num_cpus=0.5,
            encoded_batch_target_bytes=64 * MIB,
            target_min_block_size_bytes=32 * MIB,
            target_max_block_size_bytes=64 * MIB,
        ),
        output_schema=_schema(50_896),
    )

    assert tuned.config.read_concurrency == 3
    assert tuned.config.read_task_num_cpus == 1
    assert tuned.config.write_concurrency == 2
    assert tuned.config.write_task_num_cpus == 0.5
    assert tuned.config.encoded_batch_target_bytes == 64 * MIB
    assert tuned.config.target_min_block_size_bytes == 32 * MIB
    assert tuned.config.target_max_block_size_bytes == 64 * MIB


def test_profile_steps_down_batch_when_actor_memory_is_limited():
    tuned = build_cpu_iceberg_config(
        options=_options(cluster_memory_bytes=4 * 1024**3),
        output_schema=_schema(50_896),
    )
    assert tuned.config.encoded_batch_target_bytes == 128 * MIB


def test_profile_respects_explicit_unsafe_batch_instead_of_silently_changing_it():
    with pytest.raises(WorkerMemoryCapacityError):
        build_cpu_iceberg_config(
            options=_options(
                cluster_memory_bytes=4 * 1024**3,
                encoded_batch_target_bytes=256 * MIB,
            ),
            output_schema=_schema(50_896),
        )


def test_profile_propagates_explicit_resource_errors():
    with pytest.raises(ValueError, match="stage concurrency exceeds"):
        build_cpu_iceberg_config(
            options=_options(parse_actor_max_size=32, encode_actor_max_size=32),
            output_schema=_schema(50_896),
        )
