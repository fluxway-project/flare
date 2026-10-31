"""HTTP connection pooling, retry and process-safe keepalive sessions."""

import os
import socket

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.util.retry import Retry


def _build_keepalive_socket_options() -> list:
    """
    Build socket options enabling TCP keepalive so middleboxes or HTTP gateways
    do not silently reap long-lived streaming connections between intermittent reads.
    Linux-only options (TCP_KEEPIDLE/INTVL/CNT) are added best-effort.
    """
    opts = list(HTTPConnection.default_socket_options) + [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    for name, value in (
        ("TCP_KEEPIDLE", 60),
        ("TCP_KEEPINTVL", 30),
        ("TCP_KEEPCNT", 3),
    ):
        const = getattr(socket, name, None)
        if const is not None:
            opts.append((socket.IPPROTO_TCP, const, value))
    return opts


class _KeepAliveAdapter(HTTPAdapter):
    """HTTPAdapter that enables TCP SO_KEEPALIVE on every pooled connection."""

    _SOCKET_OPTIONS = _build_keepalive_socket_options()

    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = self._SOCKET_OPTIONS
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["socket_options"] = self._SOCKET_OPTIONS
        return super().proxy_manager_for(*args, **kwargs)


def create_retry_session(
    max_retries: int = 5, backoff_factor: float = 1.0
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = _KeepAliveAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Per-process Session cache.
#
# Holding a requests.Session at module scope is unsafe under fork-based
# multiprocessing: sockets sitting in the parent's connection pool would be
# inherited by every child Ray actor and the first request could fail with
# 'Bad file descriptor' / 'Connection reset' as siblings race on the same
# fd. Keying the cache by PID forces each actor process to lazily build its
# own Session on first use, while still reusing it within the process.
_SESSION_PID: int | None = None
_SESSION: requests.Session | None = None


def get_process_http_session() -> requests.Session:
    global _SESSION, _SESSION_PID
    pid = os.getpid()
    if _SESSION is None or _SESSION_PID != pid:
        _SESSION = create_retry_session()
        _SESSION_PID = pid
    return _SESSION
