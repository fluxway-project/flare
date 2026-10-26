"""Catalog construction and table-session ownership for Iceberg adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from flare.data.connectors.adapters.iceberg.catalog import (
    load_iceberg_catalog,
    normalize_iceberg_catalog_kwargs,
)
from flare.data.connectors.adapters.iceberg.identity import (
    bound_source_identity,
    require_table_uuid,
    validate_table_uuid,
)


class IcebergCatalog(Protocol):
    def load_table(self, identifier: str) -> Any: ...


@runtime_checkable
class IcebergCatalogProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def open(self) -> AbstractContextManager[IcebergCatalog]: ...


@dataclass(frozen=True, slots=True)
class ConfiguredIcebergCatalogProvider:
    catalog_kwargs: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_kwargs",
            MappingProxyType(normalize_iceberg_catalog_kwargs(self.catalog_kwargs)),
        )
        uri = self.catalog_kwargs.get("uri")
        if uri is not None:
            parsed = urlsplit(uri)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "catalog URI must not contain credentials or parameters"
                )

    @property
    def identity(self) -> str:
        # 身份只取明确的资产定位字段，不散列凭据或任意 catalog 属性。
        return json.dumps(
            {
                key: self.catalog_kwargs[key]
                for key in ("name", "type", "uri", "warehouse")
                if key in self.catalog_kwargs
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @contextmanager
    def open(self):
        catalog = load_iceberg_catalog(self.catalog_kwargs)
        try:
            yield catalog
        finally:
            catalog.close()


class IcebergTableSession:
    """One lazy catalog context and its current table, owned by one adapter."""

    def __init__(self, provider: IcebergCatalogProvider, table_identifier: str) -> None:
        self._provider = provider
        self._identifier = table_identifier
        self._context: AbstractContextManager[IcebergCatalog] | None = None
        self._catalog: IcebergCatalog | None = None
        self._table: Any = None
        self._closed = False
        self._table_uuid: str | None = None
        self._base_identity: str | None = None
        self._source_identity: str | None = None

    def table(self, *, refresh: bool = False) -> Any:
        if self._closed:
            raise RuntimeError("Iceberg table session is closed")
        if self._catalog is None:
            context = self._provider.open()
            catalog = context.__enter__()
            try:
                table = catalog.load_table(self._identifier)
                if table is None:
                    raise TypeError("Iceberg table load returned None")
                table_uuid = require_table_uuid(table)
            except BaseException as error:
                context.__exit__(type(error), error, error.__traceback__)
                raise
            self._context, self._catalog, self._table = context, catalog, table
            self._table_uuid = table_uuid
        elif refresh:
            table = self._catalog.load_table(self._identifier)
            if table is None:
                raise TypeError("Iceberg table refresh returned None")
            validate_table_uuid(table, self._table_uuid)
            self._table = table
        return self._table

    @property
    def is_bound(self) -> bool:
        return self._table_uuid is not None

    @property
    def table_uuid(self) -> str:
        if self._table_uuid is None:
            self.table()
        assert self._table_uuid is not None
        return self._table_uuid

    def source_identity(self, base_identity: str) -> str:
        if self._base_identity is not None and self._base_identity != base_identity:
            raise ValueError("Iceberg source configuration changed during session")
        if self._source_identity is None:
            self._base_identity = base_identity
            self._source_identity = bound_source_identity(
                base_identity, self.table_uuid
            )
        return self._source_identity

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        context, self._context = self._context, None
        self._catalog = self._table = None
        if context is not None:
            context.__exit__(None, None, None)
