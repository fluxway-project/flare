import pytest

from recipes.ae11.util.runtime_env import build_ae11_runtime_env


def test_cuda_runtime_is_cuda_first_and_registers_cpu_for_host_import():
    env = build_ae11_runtime_env(
        xla_cache_dir="/tmp/ae11-xla",
        platform="cuda",
        catalog_token="token",
    )

    assert env == {
        "env_vars": {
            "JAX_PLATFORMS": "cuda,cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "JAX_COMPILATION_CACHE_DIR": "/tmp/ae11-xla",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "GRAVITINO_TOKEN": "token",
        }
    }


def test_cpu_runtime_does_not_add_gpu_allocator_or_empty_token():
    env = build_ae11_runtime_env(
        xla_cache_dir="/tmp/ae11-xla",
        platform="cpu",
        catalog_token=None,
    )

    assert env["env_vars"]["JAX_PLATFORMS"] == "cpu"
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in env["env_vars"]
    assert "GRAVITINO_TOKEN" not in env["env_vars"]


def test_runtime_env_rejects_unknown_platform():
    with pytest.raises(ValueError, match="unsupported AE11 JAX platform"):
        build_ae11_runtime_env(
            xla_cache_dir="/tmp/ae11-xla",
            platform="gpu",  # type: ignore[arg-type]
            catalog_token=None,
        )
