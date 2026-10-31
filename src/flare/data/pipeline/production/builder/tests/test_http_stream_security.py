from unittest.mock import Mock

import pytest
import requests

from flare.data.pipeline.production.builder.hdfs_source import ResilientWebHdfsReader


@pytest.mark.parametrize("status", [401, None])
def test_stream_errors_do_not_expose_credentialed_urls(monkeypatch, caplog, status):
    secret = "secret-value"
    session = Mock()
    if status is None:
        session.get.side_effect = requests.ConnectionError(
            f"http://storage.invalid/?password={secret}"
        )
    else:
        response = Mock(status_code=status)
        session.get.side_effect = requests.HTTPError(
            f"http://storage.invalid/?password={secret}", response=response
        )
    monkeypatch.setattr(
        "flare.data.pipeline.production.builder.hdfs_source.get_process_http_session",
        lambda: session,
    )
    reader = ResilientWebHdfsReader(
        "/fixture/part.zst",
        url_resolver=lambda _: f"http://storage.invalid/?password={secret}",
        connect_timeout=1,
        read_timeout=1,
        max_retries=0,
    )
    with pytest.raises(RuntimeError) as error:
        reader.read(1)
    assert secret not in str(error.value) + caplog.text
