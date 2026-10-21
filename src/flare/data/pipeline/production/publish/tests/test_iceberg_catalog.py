import pytest

from flare.data.connectors.adapters.iceberg.catalog import (
    normalize_iceberg_catalog_kwargs,
)
from flare.io.aws import PYICEBERG_FSSPEC_FILE_IO


def test_rest_catalog_uses_noop_auth_and_vended_credentials() -> None:
    actual = normalize_iceberg_catalog_kwargs(
        {
            "name": "company-rest",
            "type": "rest",
            "uri": "https://catalog.example/iceberg",
            "warehouse": "dynamic-catalog",
        }
    )

    assert actual["auth"] == {"type": "noop"}
    assert actual["header.X-Iceberg-Access-Delegation"] == "vended-credentials"
    assert actual["py-io-impl"] == PYICEBERG_FSSPEC_FILE_IO


def test_rest_catalog_accepts_bearer_authentication() -> None:
    actual = normalize_iceberg_catalog_kwargs(
        {
            "name": "secured-rest",
            "type": "rest",
            "uri": "https://catalog.example/iceberg",
            "warehouse": "dynamic-catalog",
            "token": "test-token",
        }
    )

    assert actual["token"] == "test-token"
    assert "auth" not in actual


def test_rest_catalog_rejects_custom_auth_manager() -> None:
    with pytest.raises(ValueError, match="custom auth managers"):
        normalize_iceberg_catalog_kwargs(
            {
                "name": "invalid-rest",
                "type": "rest",
                "uri": "https://catalog.example/iceberg",
                "warehouse": "dynamic-catalog",
                "auth": {"type": "custom", "impl": "package.AuthManager"},
            }
        )


def test_rest_catalog_preserves_explicit_file_io_implementation() -> None:
    explicit_implementation = "package.CustomFileIO"

    actual = normalize_iceberg_catalog_kwargs(
        {
            "name": "company-rest",
            "type": "rest",
            "uri": "https://catalog.example/iceberg",
            "warehouse": "dynamic-catalog",
            "py-io-impl": explicit_implementation,
        }
    )

    assert actual["py-io-impl"] == explicit_implementation
