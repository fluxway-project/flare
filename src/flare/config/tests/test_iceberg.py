import json

import pytest

from flare.config.iceberg import (
    CatalogEndpoint,
    IcebergSettings,
    ObjectStoreSettings,
    RestAuth,
    SecretValue,
)


def test_settings_compose_rest_auth_and_vended_credentials() -> None:
    settings = IcebergSettings(
        endpoint=CatalogEndpoint(
            name="iceberg_catalog",
            catalog_type="rest",
            uri="https://catalog.example/catalog/iceberg",
            warehouse="iceberg_catalog",
        ),
        auth=RestAuth(type="bearer", token=SecretValue("secret")),
    )

    kwargs = settings.to_pyiceberg_kwargs()
    assert kwargs["token"] == "secret"
    assert "auth" not in kwargs
    assert kwargs["header.X-Iceberg-Access-Delegation"] == "vended-credentials"
    assert "token=SecretValue(<redacted>)" in repr(settings)


def test_non_iceberg_route_is_not_an_iceberg_endpoint() -> None:
    with pytest.raises(ValueError, match="/iceberg endpoint"):
        CatalogEndpoint(
            name="catalog",
            catalog_type="rest",
            uri="https://catalog.example/not-iceberg",
            warehouse="warehouse",
        )


def test_object_store_rejects_mixed_profile_and_keys() -> None:
    with pytest.raises(ValueError, match="profile cannot be combined"):
        ObjectStoreSettings.from_options(
            {
                "profile": "default",
                "access_key": "access",
                "secret_key": "secret",
            }
        )


def test_json_config_resolves_environment_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_GRAVITINO_TOKEN", "secret-from-env")
    path = tmp_path / "iceberg.json"
    path.write_text(
        json.dumps(
            {
                "endpoint": {
                    "name": "catalog",
                    "type": "rest",
                    "uri": "https://catalog.example/iceberg",
                    "warehouse": "warehouse",
                },
                "auth": {"type": "bearer", "token": "${TEST_GRAVITINO_TOKEN}"},
            }
        ),
        encoding="utf-8",
    )

    settings = IcebergSettings.from_file(path)
    assert settings.to_pyiceberg_kwargs()["token"] == "secret-from-env"


def test_toml_config_resolves_environment_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_GRAVITINO_TOKEN", "secret-from-env-toml")
    path = tmp_path / "iceberg.toml"
    path.write_text(
        """
[endpoint]
name = "catalog"
type = "rest"
uri = "https://catalog.example/iceberg"
warehouse = "warehouse"

[auth]
type = "bearer"
token = "${TEST_GRAVITINO_TOKEN}"
""",
        encoding="utf-8",
    )

    settings = IcebergSettings.from_file(path)
    assert settings.to_pyiceberg_kwargs()["token"] == "secret-from-env-toml"
