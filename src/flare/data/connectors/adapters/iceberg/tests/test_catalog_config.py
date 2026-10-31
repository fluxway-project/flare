"""Shared pipeline contracts, independent of recipe configurations."""

from __future__ import annotations

import pytest


class TestIcebergCatalogConfig:
    def test_rest_catalog_requires_uri(self):
        from flare.data.connectors.adapters.iceberg.catalog import (
            normalize_iceberg_catalog_kwargs,
        )

        with pytest.raises(ValueError, match="REST Iceberg catalog requires"):
            normalize_iceberg_catalog_kwargs({"name": "rest", "type": "rest"})

    def test_catalog_type_is_preserved_for_the_native_loader(self):
        from flare.data.connectors.adapters.iceberg.catalog import (
            normalize_iceberg_catalog_kwargs,
        )

        config = normalize_iceberg_catalog_kwargs(
            {
                "name": "local",
                "type": "hadoop",
                "warehouse": "file:///tmp/iceberg_warehouse",
            }
        )

        assert config["type"] == "hadoop"

    def test_in_memory_catalog_requires_warehouse(self):
        from flare.data.connectors.adapters.iceberg.catalog import (
            normalize_iceberg_catalog_kwargs,
        )

        with pytest.raises(ValueError, match="requires 'warehouse'"):
            normalize_iceberg_catalog_kwargs({"name": "local", "type": "in-memory"})
