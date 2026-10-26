from types import SimpleNamespace

import pytest

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
    IcebergTableSession,
)
from flare.data.connectors.adapters.iceberg.tests.catalog_fixtures import (
    MemoryCatalogProvider,
)


def test_table_access_reuses_catalog_until_explicit_refresh_and_closes_once():
    tables = iter(
        SimpleNamespace(metadata=SimpleNamespace(table_uuid="uuid")) for _ in range(2)
    )
    provider = MemoryCatalogProvider(lambda _identifier: next(tables))
    session = IcebergTableSession(provider, "namespace.table")
    first = session.table()
    assert all(session.table() is first for _ in range(20))
    assert session.table(refresh=True) is not first
    assert provider.opens == 1
    session.close()
    session.close()
    assert provider.closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.table()


def test_initial_load_error_releases_catalog_context():
    def load(_identifier):
        raise ValueError("metadata unavailable")

    provider = MemoryCatalogProvider(load)
    with pytest.raises(ValueError, match="metadata unavailable"):
        IcebergTableSession(provider, "namespace.table").table()
    assert provider.opens == provider.closes == 1


def test_catalog_identity_never_hashes_arbitrary_credentials():
    shared = {"name": "catalog", "type": "rest", "uri": "http://catalog"}
    first = ConfiguredIcebergCatalogProvider(
        {**shared, "s3.secret-access-key": "first", "token": "first"}
    )
    second = ConfiguredIcebergCatalogProvider(
        {**shared, "s3.secret-access-key": "second", "token": "second"}
    )
    assert first.identity == second.identity
    assert "first" not in first.identity
