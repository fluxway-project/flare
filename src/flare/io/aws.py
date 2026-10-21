from __future__ import annotations

import dataclasses
import json
import os

from pyarrow import fs
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.utils.properties import get_first_property_value


class SafeFsspecFileIO(FsspecFileIO):
    def __init__(self, properties: dict | None = None) -> None:
        super().__init__({} if properties is None else properties)
        self._scheme_to_fs["s3"] = _custom_s3
        self._scheme_to_fs["s3a"] = _custom_s3
        self._scheme_to_fs["s3n"] = _custom_s3


def _custom_s3(properties: dict):
    from s3fs import S3FileSystem

    return S3FileSystem(
        anon=False,
        key=get_first_property_value(
            properties, "s3.access-key-id", "client.credentials.access-key-id"
        ),
        secret=get_first_property_value(
            properties,
            "s3.secret-access-key",
            "client.credentials.secret-access-key",
        ),
        token=get_first_property_value(
            properties, "s3.session-token", "client.credentials.session-token"
        ),
        client_kwargs={
            "endpoint_url": properties.get("s3.endpoint"),
            "region_name": get_first_property_value(
                properties, "s3.region", "client.region"
            ),
        },
        config_kwargs={},
    )


PYICEBERG_FSSPEC_FILE_IO = "flare.io.aws.SafeFsspecFileIO"
_S3_OPTION_KEYS = frozenset(
    {"endpoint", "access_key", "secret_key", "session_token", "region", "profile"}
)


@dataclasses.dataclass
class S3Config:
    s3_endpoint: str | None = None
    s3_region: str | None = None
    s3_profile: str | None = None
    s3_config: str | None = None

    def to_options(self) -> dict:
        options: dict = {}
        if self.s3_config:
            config_path = self.s3_config
            if not os.path.isfile(config_path):
                raise FileNotFoundError(config_path)
            with open(config_path, encoding="utf-8") as source:
                options.update(json.load(source))
        if self.s3_endpoint:
            options["endpoint"] = self.s3_endpoint
        if self.s3_region:
            options["region"] = self.s3_region
        if self.s3_profile:
            options["profile"] = self.s3_profile
        environment = {
            "endpoint": "AWS_ENDPOINT_URL",
            "access_key": "AWS_ACCESS_KEY_ID",
            "secret_key": "AWS_SECRET_ACCESS_KEY",
            "session_token": "AWS_SESSION_TOKEN",
            "region": "AWS_REGION",
            "profile": "AWS_PROFILE",
        }
        for option, variable in environment.items():
            if option not in options and os.environ.get(variable):
                options[option] = os.environ[variable]
        return normalize_s3_options(options)


def normalize_s3_options(s3_options: dict | None) -> dict:
    if not s3_options:
        return {}
    unknown = sorted(set(s3_options) - _S3_OPTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown S3 option keys: {unknown}")
    normalized = {
        key: value
        for key, value in s3_options.items()
        if value is not None and value != ""
    }
    has_access = "access_key" in normalized
    has_secret = "secret_key" in normalized
    if has_access != has_secret:
        raise ValueError("S3 access_key and secret_key must be configured together")
    if "session_token" in normalized and not has_access:
        raise ValueError("S3 session_token requires access_key and secret_key")
    if "profile" in normalized and (
        has_access or has_secret or "session_token" in normalized
    ):
        raise ValueError("S3 profile cannot be combined with explicit access keys")
    return normalized


def get_vortex_storage_options(s3_options: dict) -> dict:
    options = normalize_s3_options(s3_options)
    result: dict = {}
    names = {
        "endpoint": "aws_endpoint",
        "access_key": "aws_access_key_id",
        "secret_key": "aws_secret_access_key",
        "session_token": "aws_session_token",
        "region": "aws_region",
        "profile": "aws_profile",
    }
    result.update({names[key]: value for key, value in options.items()})
    if "endpoint" in options:
        result["aws_virtual_hosted_style_request"] = "false"
        if options["endpoint"].startswith("http://"):
            result["aws_allow_http"] = "true"
    return result


def get_pyiceberg_s3_kwargs(s3_options: dict) -> dict:
    options = normalize_s3_options(s3_options)
    if "profile" in options:
        raise ValueError("PyIceberg PyArrowFileIO does not support S3 profile")
    result = {"py-io-impl": PYICEBERG_FSSPEC_FILE_IO}
    names = {
        "endpoint": "s3.endpoint",
        "access_key": "s3.access-key-id",
        "secret_key": "s3.secret-access-key",
        "session_token": "s3.session-token",
        "region": "s3.region",
    }
    result.update({names[key]: value for key, value in options.items() if key in names})
    if "endpoint" in options:
        result["s3.path-style-access"] = "true"
    return result


def get_pyarrow_s3_fs(s3_options: dict) -> fs.FileSystem | None:
    options = normalize_s3_options(s3_options)
    if not options:
        return None
    kwargs: dict = {}
    if "endpoint" in options:
        kwargs["endpoint_override"] = options["endpoint"]
        kwargs["force_virtual_addressing"] = False
    names = {
        "access_key": "access_key",
        "secret_key": "secret_key",
        "session_token": "session_token",
        "region": "region",
    }
    kwargs.update({names[key]: value for key, value in options.items() if key in names})
    return fs.S3FileSystem(**kwargs)


__all__ = [
    "PYICEBERG_FSSPEC_FILE_IO",
    "S3Config",
    "SafeFsspecFileIO",
    "get_pyarrow_s3_fs",
    "get_pyiceberg_s3_kwargs",
    "get_vortex_storage_options",
    "normalize_s3_options",
]
