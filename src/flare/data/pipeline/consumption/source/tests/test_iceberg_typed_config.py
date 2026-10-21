from __future__ import annotations

import json

import pytest

from flare.data.pipeline.consumption.bootstrap.factory import iceberg_catalog_kwargs


def test_iceberg_config_resolves_bearer_token_from_worker_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TEST_WORKER_GRAVITINO_TOKEN", "worker-token")
    config_path = tmp_path / "iceberg.json"
    config_path.write_text(
        json.dumps(
            {
                "endpoint": {
                    "name": "catalog",
                    "type": "rest",
                    "uri": "https://catalog.example/iceberg",
                    "warehouse": "warehouse",
                },
                "auth": {
                    "type": "bearer",
                    "token": "${TEST_WORKER_GRAVITINO_TOKEN}",
                },
            }
        ),
        encoding="utf-8",
    )

    actual = iceberg_catalog_kwargs({"iceberg_config": config_path})

    assert actual["token"] == "worker-token"
    assert actual["header.X-Iceberg-Access-Delegation"] == "vended-credentials"


def test_iceberg_config_rejects_individual_catalog_arguments(tmp_path) -> None:
    config_path = tmp_path / "iceberg.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined"):
        iceberg_catalog_kwargs(
            {"iceberg_config": config_path, "catalog_name": "catalog"}
        )
