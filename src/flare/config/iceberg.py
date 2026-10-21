from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlparse

from flare.io.aws import get_pyiceberg_s3_kwargs, normalize_s3_options

_ENV_PATTERN = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
AuthType = Literal["noop", "bearer", "basic", "oauth2"]


@dataclasses.dataclass(frozen=True, slots=True)
class SecretValue:
    _value: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if not self._value:
            raise ValueError("secret value must not be empty")

    @classmethod
    def from_environment(cls, name: str) -> SecretValue:
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"missing secret environment variable: {name}")
        return cls(value)

    @property
    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretValue(<redacted>)"


@dataclasses.dataclass(frozen=True, slots=True)
class CatalogEndpoint:
    name: str
    catalog_type: str
    uri: str
    warehouse: str

    def __post_init__(self) -> None:
        for field_name in ("name", "catalog_type", "uri", "warehouse"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"catalog endpoint {field_name} must be non-empty")
        if self.catalog_type != "rest":
            raise ValueError(f"unsupported catalog type: {self.catalog_type!r}")
        endpoint_path = urlparse(self.uri.rstrip("/")).path.rstrip("/")
        if not endpoint_path or endpoint_path.rsplit("/", 1)[-1] != "iceberg":
            raise ValueError("Iceberg REST URI must include the /iceberg endpoint")


@dataclasses.dataclass(frozen=True, slots=True)
class RestAuth:
    type: AuthType
    token: SecretValue | None = None
    username: str | None = None
    password: SecretValue | None = None
    client_id: str | None = None
    client_secret: SecretValue | None = None
    token_url: str | None = None
    scope: str | None = None

    def __post_init__(self) -> None:
        if self.type == "noop":
            if any(
                value is not None
                for value in (
                    self.token,
                    self.username,
                    self.password,
                    self.client_id,
                    self.client_secret,
                    self.token_url,
                    self.scope,
                )
            ):
                raise ValueError("noop auth cannot contain credentials")
        elif self.type == "bearer":
            if self.token is None or any(
                value is not None
                for value in (
                    self.username,
                    self.password,
                    self.client_id,
                    self.client_secret,
                    self.token_url,
                )
            ):
                raise ValueError("bearer auth requires only token")
        elif self.type == "basic":
            if not self.username or self.password is None:
                raise ValueError("basic auth requires username and password")
        elif self.type == "oauth2":
            if not self.client_id or self.client_secret is None or not self.token_url:
                raise ValueError(
                    "oauth2 auth requires client_id, client_secret and token_url"
                )
        else:
            raise ValueError(f"unsupported REST auth type: {self.type!r}")

    @classmethod
    def noop(cls) -> RestAuth:
        return cls(type="noop")

    @classmethod
    def bearer_from_environment(cls, name: str) -> RestAuth:
        return cls(type="bearer", token=SecretValue.from_environment(name))

    def to_pyiceberg(self) -> dict[str, Any]:
        if self.type == "noop":
            return {"auth": {"type": "noop"}}
        if self.type == "bearer":
            return {"token": self.token.value}  # type: ignore[union-attr]
        if self.type == "basic":
            return {
                "auth": {
                    "type": "basic",
                    "basic": {
                        "username": self.username,
                        "password": self.password.value,  # type: ignore[union-attr]
                    },
                }
            }
        return {
            "auth": {
                "type": "oauth2",
                "oauth2": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret.value,  # type: ignore[union-attr]
                    "token_url": self.token_url,
                    **({"scope": self.scope} if self.scope else {}),
                },
            }
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectStoreSettings:
    options: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.options, MappingProxyType):
            raise TypeError("object store options must be a MappingProxyType")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> ObjectStoreSettings:
        return cls(options=MappingProxyType(normalize_s3_options(dict(options))))

    def to_pyiceberg(self) -> dict[str, Any]:
        return get_pyiceberg_s3_kwargs(dict(self.options)) if self.options else {}


@dataclasses.dataclass(frozen=True, slots=True)
class IcebergSettings:
    endpoint: CatalogEndpoint
    auth: RestAuth
    object_store: ObjectStoreSettings = dataclasses.field(
        default_factory=ObjectStoreSettings, repr=False
    )
    access_delegation: Literal["vended-credentials"] = "vended-credentials"
    ssl: Mapping[str, Any] | None = dataclasses.field(default=None, repr=False)
    headers: Mapping[str, str] = dataclasses.field(default_factory=dict, repr=False)

    def to_pyiceberg_kwargs(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.endpoint.name,
            "type": self.endpoint.catalog_type,
            "uri": self.endpoint.uri,
            "warehouse": self.endpoint.warehouse,
            "header.X-Iceberg-Access-Delegation": self.access_delegation,
            **self.auth.to_pyiceberg(),
            **self.object_store.to_pyiceberg(),
        }
        if self.ssl is not None:
            result["ssl"] = dict(self.ssl)
        result.update({f"header.{key}": value for key, value in self.headers.items()})
        return result

    @classmethod
    def from_file(cls, path: Path) -> IcebergSettings:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as source:
            if path.suffix == ".json":
                raw = json.load(source)
            elif path.suffix == ".toml":
                import tomllib

                raw = tomllib.load(source)
            else:
                raise ValueError("Iceberg config must be a .json or .toml file")
        if not isinstance(raw, Mapping):
            raise TypeError("Iceberg config root must be an object")
        return cls.from_mapping(_resolve_environment(raw))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> IcebergSettings:
        endpoint = raw.get("endpoint")
        auth = raw.get("auth")
        if not isinstance(endpoint, Mapping) or not isinstance(auth, Mapping):
            raise TypeError("Iceberg config requires endpoint and auth objects")
        auth_type = auth.get("type")
        if auth_type == "noop":
            rest_auth = RestAuth.noop()
        elif auth_type == "bearer":
            rest_auth = RestAuth(type="bearer", token=SecretValue(auth["token"]))
        elif auth_type == "basic":
            rest_auth = RestAuth(
                type="basic",
                username=auth["username"],
                password=SecretValue(auth["password"]),
            )
        elif auth_type == "oauth2":
            rest_auth = RestAuth(
                type="oauth2",
                client_id=auth["client_id"],
                client_secret=SecretValue(auth["client_secret"]),
                token_url=auth["token_url"],
                scope=auth.get("scope"),
            )
        else:
            raise ValueError(f"unsupported auth.type: {auth_type!r}")
        object_store = raw.get("object_store", {})
        if not isinstance(object_store, Mapping):
            raise TypeError("object_store must be an object")
        return cls(
            endpoint=CatalogEndpoint(
                name=endpoint["name"],
                catalog_type=endpoint.get("type", "rest"),
                uri=endpoint["uri"],
                warehouse=endpoint["warehouse"],
            ),
            auth=rest_auth,
            object_store=ObjectStoreSettings.from_options(object_store),
            access_delegation=raw.get("access_delegation", "vended-credentials"),
            ssl=raw.get("ssl"),
            headers=raw.get("headers", {}),
        )


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.fullmatch(value)
        return os.environ[match.group(1)] if match else value
    return value


__all__ = [
    "CatalogEndpoint",
    "IcebergSettings",
    "ObjectStoreSettings",
    "RestAuth",
    "SecretValue",
]
