from types import SimpleNamespace

import pytest

from flare.io.aws import (
    get_pyarrow_s3_fs,
    get_pyiceberg_s3_kwargs,
    normalize_s3_options,
)


def test_pyiceberg_s3_kwargs_use_pyarrow_file_io():
    kwargs = get_pyiceberg_s3_kwargs(
        {
            "endpoint": "http://s3.local:9000",
            "access_key": "ak",
            "secret_key": "sk",
            "region": "us-east-1",
        }
    )

    assert kwargs == {
        "py-io-impl": "flare.io.aws.SafeFsspecFileIO",
        "s3.endpoint": "http://s3.local:9000",
        "s3.path-style-access": "true",
        "s3.access-key-id": "ak",
        "s3.secret-access-key": "sk",
        "s3.region": "us-east-1",
    }


def test_pyiceberg_s3_kwargs_allow_private_endpoint_without_region():
    kwargs = get_pyiceberg_s3_kwargs(
        {
            "endpoint": "http://s3.local:9000",
            "access_key": "ak",
            "secret_key": "sk",
        }
    )

    assert kwargs == {
        "py-io-impl": "flare.io.aws.SafeFsspecFileIO",
        "s3.endpoint": "http://s3.local:9000",
        "s3.path-style-access": "true",
        "s3.access-key-id": "ak",
        "s3.secret-access-key": "sk",
    }


def test_temporary_s3_credentials_include_session_token(monkeypatch):
    observed = {}

    monkeypatch.setattr(
        "pyarrow.fs.S3FileSystem",
        lambda **kwargs: observed.update(kwargs) or SimpleNamespace(),
    )
    options = {
        "endpoint": "http://s3.local:9000",
        "access_key": "ak",
        "secret_key": "sk",
        "session_token": "token",
        "region": "us-east-1",
    }

    get_pyarrow_s3_fs(options)

    assert observed["session_token"] == "token"
    assert get_pyiceberg_s3_kwargs(options)["s3.session-token"] == "token"


def test_s3_options_drop_absent_values():
    assert normalize_s3_options(
        {
            "endpoint": None,
            "access_key": "",
            "secret_key": None,
            "region": "us-east-1",
        }
    ) == {"region": "us-east-1"}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"access_key": "ak"}, "access_key and secret_key"),
        ({"session_token": "token"}, "session_token requires access_key"),
        (
            {"access_key": "ak", "secret_key": "sk", "profile": "profile"},
            "profile cannot be combined",
        ),
        ({"aws_access_key_id": "ak"}, "Unknown S3 option"),
    ],
)
def test_s3_options_reject_invalid_config(options, message):
    with pytest.raises(ValueError, match=message):
        normalize_s3_options(options)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"profile": "profile"}, "does not support S3 profile"),
    ],
)
def test_pyiceberg_s3_kwargs_reject_unsupported_config(options, message):
    with pytest.raises(ValueError, match=message):
        get_pyiceberg_s3_kwargs(options)
