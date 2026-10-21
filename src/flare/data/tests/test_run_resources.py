from pathlib import Path

import pytest
import ray

from flare.data import run
from flare.data.pipeline.production.publish.runner import _initialize_ray
from flare.runtime.ray import RayConfig


def test_auto_ray_connection_failure_has_no_local_fallback(monkeypatch):
    calls = []

    def fail_connection(*, address, **kwargs):
        calls.append(address)
        raise ConnectionError("cluster unavailable")

    monkeypatch.setattr(ray, "init", fail_connection)
    args = run.RunConfig(
        output="namespace.table",
        output_format="iceberg",
        input="/tmp/input",
        conf_dir=Path("/tmp/conf"),
        model_conf_dir=Path("/tmp/model"),
        ray=RayConfig(ray_address="auto"),
    )

    with pytest.raises(ConnectionError, match="cluster unavailable"):
        _initialize_ray(args)

    assert calls == ["auto"]


def test_hdfs_mode_requires_explicit_publication_window() -> None:
    with pytest.raises(ValueError, match="publication-window-files"):
        run.RunConfig(
            output="namespace.table",
            output_format="iceberg",
            hdfs_root="/data",
            conf_dir=Path("/tmp/conf"),
            model_conf_dir=Path("/tmp/model"),
        )


def test_commit_layout_observation_defaults_to_disabled() -> None:
    config = run.RunConfig(
        output="namespace.table",
        output_format="iceberg",
        input="/tmp/input",
        conf_dir=Path("/tmp/conf"),
        model_conf_dir=Path("/tmp/model"),
    )

    assert not config.iceberg_commit_layout_observation
