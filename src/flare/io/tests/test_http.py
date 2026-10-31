import socket
from unittest.mock import Mock

from flare.io import http


def test_retry_session_uses_same_retry_and_keepalive_policy_for_both_schemes():
    with http.create_retry_session(max_retries=2, backoff_factor=0.5) as session:
        adapter = session.get_adapter("https://example.invalid/")
        assert adapter is session.get_adapter("http://example.invalid/")
        assert adapter.max_retries.total == 2
        assert adapter.max_retries.connect == 2
        assert adapter.max_retries.read == 2
        assert adapter.max_retries.backoff_factor == 0.5
        assert (
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1,
        ) in adapter.poolmanager.connection_pool_kw["socket_options"]


def test_session_cache_is_reused_only_within_its_process(monkeypatch):
    factory = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(http, "create_retry_session", factory)
    monkeypatch.setattr(http, "_SESSION", None)
    monkeypatch.setattr(http, "_SESSION_PID", None)
    pid = [100]
    monkeypatch.setattr(http.os, "getpid", lambda: pid[0])
    first = http.get_process_http_session()
    assert http.get_process_http_session() is first
    pid[0] = 101
    assert http.get_process_http_session() is not first
    assert factory.call_count == 2
