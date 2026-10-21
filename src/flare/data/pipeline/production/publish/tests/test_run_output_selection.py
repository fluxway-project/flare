from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import tyro

import flare.data.pipeline.production.publish.runner as publish_runner
from flare.data.connectors.adapters.parquet.sink import ParquetDatasetPublisher
from flare.data.pipeline.production.publish.runner import (
    _build_output_publisher,
    _resolve_feature_manifest_dir,
    _resolve_output_path,
    _resolve_resume_state_path,
    _validate_output_arguments,
)
from flare.data.run import RunConfig


def _args(tmp_path, **overrides):
    values = {
        "output": str(tmp_path / "dataset"),
        "output_format": "parquet",
        "overwrite": False,
        "iceberg_catalog_name": None,
        "iceberg_catalog_type": None,
        "iceberg_catalog_uri": None,
        "iceberg_catalog_warehouse": None,
        "iceberg_writer_target_size_mib": None,
        "iceberg_commit_layout_observation": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Pipeline:
    output_schema = pa.schema(
        [
            pa.field("fid_101", pa.int64()),
            pa.field("_ds", pa.string()),
            pa.field("_hour", pa.string()),
        ]
    )
    config = SimpleNamespace(
        target_max_block_size_bytes=1024,
        write_task_num_cpus=1.0,
        write_concurrency=1,
    )


def test_output_format_is_required() -> None:
    with pytest.raises(SystemExit):
        tyro.cli(RunConfig, args=["--output", "namespace.table"])


def test_local_parquet_output_is_explicit_and_self_describing(tmp_path) -> None:
    args = _args(tmp_path)
    _validate_output_arguments(args, storage_options={})
    args.output = _resolve_output_path(args.output, args.output_format)

    publisher = _build_output_publisher(
        args,
        catalog_kwargs={},
        storage_options={},
        pipeline=_Pipeline(),
    )

    assert isinstance(publisher, ParquetDatasetPublisher)
    assert publisher.root == tmp_path / "dataset"
    assert _resolve_feature_manifest_dir(args.output, args.output_format) == str(
        tmp_path / "dataset"
    )


def test_local_output_resume_state_stays_in_canonical_state_root(tmp_path) -> None:
    args = _args(tmp_path)
    args.output = str((tmp_path / "dataset").resolve())
    args.resume_state_path = None

    path = _resolve_resume_state_path(args)

    assert Path(path).parent == (Path("data") / "_resume").resolve()


def test_iceberg_layout_observation_is_forwarded_to_publisher(
    tmp_path, monkeypatch
) -> None:
    args = _args(
        tmp_path,
        output="catalog.db.table",
        output_format="iceberg",
        iceberg_commit_layout_observation=True,
    )

    captured = {}

    class _Publisher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(publish_runner, "IcebergSnapshotPublisher", _Publisher)

    publisher = publish_runner._build_output_publisher(
        args,
        catalog_kwargs={"type": "memory"},
        storage_options={},
        pipeline=_Pipeline(),
    )

    assert isinstance(publisher, _Publisher)
    assert captured["observe_file_layout"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"iceberg_catalog_name": "catalog"},
        {"iceberg_catalog_type": "rest"},
        {"iceberg_catalog_uri": "http://catalog"},
        {"iceberg_catalog_warehouse": "warehouse"},
    ],
)
def test_local_parquet_rejects_iceberg_only_arguments(tmp_path, overrides) -> None:
    with pytest.raises(ValueError, match="Parquet output"):
        _validate_output_arguments(_args(tmp_path, **overrides), storage_options={})


def test_local_parquet_rejects_s3_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="Parquet output"):
        _validate_output_arguments(
            _args(tmp_path),
            storage_options={"endpoint": "http://s3"},
        )
