"""Native filesystem discovery and resumable HTTP streams.

Filesystem URIs use PyArrow's standard filesystem implementations. Custom
transports supply source providers or explicit listing/URL resolver callbacks;
no deployment-specific gateway discovery or authentication is performed here.
"""

import concurrent.futures
import glob
import io
import itertools
import logging
import os
import posixpath
import random
import re
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import requests
import urllib3

from flare.io.http import get_process_http_session

logger = logging.getLogger(__name__)


_STREAM_CHUNK = 4 << 20  # 4 MiB — 减少 syscall 次数，提升解压吞吐


@dataclass(frozen=True, slots=True)
class HdfsFileObservation:
    path: str
    length: int
    modification_time_ms: int
    is_directory: bool

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("HDFS observation path must not be empty")
        if self.length < 0:
            raise ValueError("HDFS observation length must be non-negative")
        if self.modification_time_ms < 0:
            raise ValueError("HDFS observation modification time must be non-negative")


@dataclass(frozen=True, slots=True)
class HdfsPartitionCursor:
    partition: str
    fingerprint: tuple[tuple[str, int, int, bool], ...] = ()
    observed_at_ms: int | None = None
    published_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.partition:
            raise ValueError("HDFS partition must not be empty")
        if self.observed_at_ms is not None and self.observed_at_ms < 0:
            raise ValueError("HDFS cursor observed_at_ms must be non-negative")

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "fingerprint": [list(item) for item in self.fingerprint],
            "observed_at_ms": self.observed_at_ms,
            "published_paths": list(self.published_paths),
        }

    @classmethod
    def from_checkpoint(cls, state: Mapping[str, object]) -> "HdfsPartitionCursor":
        expected_fields = {
            "partition",
            "fingerprint",
            "observed_at_ms",
            "published_paths",
        }
        if set(state) != expected_fields:
            raise ValueError(
                "HDFS partition cursor fields do not match protocol: "
                f"expected={sorted(expected_fields)} actual={sorted(state)}"
            )
        partition = state["partition"]
        if not isinstance(partition, str) or not partition:
            raise TypeError("HDFS cursor partition must be a non-empty string")
        raw_fingerprint = state["fingerprint"]
        if not isinstance(raw_fingerprint, list):
            raise TypeError("HDFS cursor fingerprint must be a list")
        fingerprint = []
        for item in raw_fingerprint:
            if not isinstance(item, list) or len(item) != 4:
                raise TypeError(
                    "HDFS cursor fingerprint entry must be a four-item list"
                )
            path, length, modification_time_ms, is_directory = item
            if not isinstance(path, str) or not path:
                raise TypeError("HDFS cursor fingerprint path must be non-empty")
            if not isinstance(length, int) or isinstance(length, bool) or length < 0:
                raise TypeError("HDFS cursor fingerprint length must be non-negative")
            if (
                not isinstance(modification_time_ms, int)
                or isinstance(modification_time_ms, bool)
                or modification_time_ms < 0
            ):
                raise TypeError(
                    "HDFS cursor fingerprint modification time must be non-negative"
                )
            if not isinstance(is_directory, bool):
                raise TypeError(
                    "HDFS cursor fingerprint directory flag must be boolean"
                )
            fingerprint.append((path, length, modification_time_ms, is_directory))
        raw_published = state["published_paths"]
        if not isinstance(raw_published, list):
            raise TypeError("HDFS cursor published paths must be a list")
        if any(not isinstance(path, str) or not path for path in raw_published):
            raise TypeError("HDFS cursor published paths must be non-empty strings")
        observed = state["observed_at_ms"]
        if observed is not None and (
            not isinstance(observed, int) or isinstance(observed, bool) or observed < 0
        ):
            raise TypeError("HDFS cursor observed_at_ms must be non-negative")
        return cls(
            partition=partition,
            fingerprint=tuple(fingerprint),
            observed_at_ms=observed,
            published_paths=tuple(raw_published),
        )


def _observations_fingerprint(
    observations: tuple[HdfsFileObservation, ...],
) -> tuple[tuple[str, int, int, bool], ...]:
    return tuple(
        (item.path, item.length, item.modification_time_ms, item.is_directory)
        for item in sorted(observations, key=lambda value: value.path)
    )


def _normalize_hdfs_path(path: str) -> str:
    if path.startswith("hdfs://"):
        scheme = "hdfs://"
        rest = path[7:]
    else:
        scheme = ""
        rest = path
    return scheme + re.sub(r"//+", "/", rest)


def open_hdfs_file(path: str):
    """Open a file using the native filesystem selected by its URI."""
    from pyarrow import fs

    filesystem, native_path = fs.FileSystem.from_uri(path)
    return filesystem.open_input_file(native_path)


def hdfs_ls(hdfs_dir: str) -> list[str]:
    """List direct children through the URI's native filesystem."""
    return [item.path for item in hdfs_list_status(hdfs_dir)]


def hdfs_list_status(hdfs_dir: str) -> tuple[HdfsFileObservation, ...]:
    """Read native filesystem metadata without opening any data file."""
    from pyarrow import fs

    filesystem, native_path = fs.FileSystem.from_uri(hdfs_dir)
    entries = filesystem.get_file_info(fs.FileSelector(native_path, recursive=False))
    parsed = urllib.parse.urlsplit(hdfs_dir)
    observations = []
    for entry in entries:
        path = entry.path
        if parsed.scheme:
            relative = posixpath.relpath(entry.path, native_path)
            path = hdfs_dir.rstrip("/") + "/" + urllib.parse.quote(relative, safe="/")
        is_directory = entry.type == fs.FileType.Directory
        observations.append(
            HdfsFileObservation(
                path=path,
                length=0 if is_directory else entry.size,
                modification_time_ms=(entry.mtime_ns or 0) // 1_000_000,
                is_directory=is_directory,
            )
        )
    return tuple(sorted(observations, key=lambda item: item.path))


def hdfs_file_exists(path: str) -> bool:
    normalized = _normalize_hdfs_path(path)
    parent, _, name = normalized.rpartition("/")
    if not parent or not name:
        raise ValueError(f"HDFS file path must include a parent and name: {path!r}")
    return any(
        not item.is_directory and _normalize_hdfs_path(item.path) == normalized
        for item in hdfs_list_status(parent)
    )


def observe_hdfs_partition(
    partition_path: str,
    *,
    now_ms: int | None = None,
    cursor: HdfsPartitionCursor | None = None,
    stability_window_s: float | None = None,
) -> tuple[tuple[HdfsFileObservation, ...], HdfsPartitionCursor, bool]:
    """Observe a partition and decide whether its namespace is publishable."""
    if stability_window_s is not None and stability_window_s <= 0:
        raise ValueError("stability_window_s must be positive when configured")
    observations = list(hdfs_list_status(partition_path))
    hour_dirs = [item for item in observations if item.is_directory]
    for hour in hour_dirs:
        observations.extend(
            item for item in hdfs_list_status(hour.path) if not item.is_directory
        )
    files = tuple(item for item in observations if not item.is_directory)
    current_fingerprint = _observations_fingerprint(files)
    import time as _time

    observed_at = int(_time.time() * 1000) if now_ms is None else int(now_ms)
    previous = cursor or HdfsPartitionCursor(partition=partition_path)
    previous_observed_at_ms = previous.observed_at_ms
    unchanged = (
        previous.fingerprint == current_fingerprint
        and previous_observed_at_ms is not None
    )
    stable = (
        unchanged
        and stability_window_s is not None
        and observed_at - previous_observed_at_ms >= stability_window_s * 1000
    )
    has_success = any(item.path.rsplit("/", 1)[-1] == "_SUCCESS" for item in files)
    complete = has_success or stable
    next_cursor = HdfsPartitionCursor(
        partition=partition_path,
        fingerprint=current_fingerprint,
        observed_at_ms=previous.observed_at_ms if unchanged else observed_at,
        published_paths=previous.published_paths,
    )
    return files, next_cursor, complete


def hdfs_ls_parallel(hdfs_dirs: list[str], max_workers: int = 24) -> list[str]:
    """
    Parallelize hdfs_ls across multiple directories using a thread pool.
    Returns a flat list of all absolute HDFS paths found.
    Any listing failure is propagated; an incomplete namespace observation must
    never be treated as a completed partition.
    """
    return list(hdfs_ls_parallel_iter(hdfs_dirs, max_workers=max_workers))


def hdfs_ls_parallel_iter(
    hdfs_dirs: list[str],
    max_workers: int = 24,
    *,
    list_directory: Callable[[str], list[str]] | None = None,
) -> Iterator[str]:
    """List directories concurrently and yield entries in input order."""
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")
    list_directory = hdfs_ls if list_directory is None else list_directory
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(list_directory, directory) for directory in hdfs_dirs
        ]
        for future in futures:
            yield from sorted(future.result())


class ResilientWebHdfsReader(io.RawIOBase):
    """
    File-like object that streams from WebHDFS and transparently resumes
    from the last read byte offset upon encountering network disruptions.
    """

    def __init__(
        self,
        hdfs_file: str,
        url_resolver: Callable[[str], str],
        connect_timeout: int,
        read_timeout: int,
        max_retries: int = 5,
    ):
        self.hdfs_file = hdfs_file
        self.url_resolver = url_resolver
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self.bytes_read = 0
        self.resp = None
        self.stream = None

    def readable(self) -> bool:
        return True

    def _close_response(self, reason: str):
        if self.resp:
            logger.warning(
                "Closing HDFS stream for %s at offset %d (Reason: %s)",
                self.hdfs_file,
                self.bytes_read,
                reason,
            )
            try:
                self.resp.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self.resp = None
            self.stream = None

    def _open_stream(self):
        self._close_response("Reopening stream")

        url = self.url_resolver(self.hdfs_file)
        if self.bytes_read > 0:
            url += f"&offset={self.bytes_read}"
            logger.info(
                "Resuming HDFS stream for %s from offset %d",
                self.hdfs_file,
                self.bytes_read,
            )
        else:
            logger.info("Opening HDFS stream for %s", self.hdfs_file)

        self.resp = get_process_http_session().get(
            url, stream=True, timeout=(self.connect_timeout, self.read_timeout)
        )
        self.resp.raise_for_status()
        self.resp.raw.decode_content = False
        self.stream = self.resp.raw

    # Status codes that ARE transient and worth retrying. Everything else in
    # the 4xx range (auth/missing/bad-request) signals a deterministic client
    # error: retrying just wastes the budget and produces a misleading
    # 'max retries exceeded' message instead of surfacing the real cause.
    _RETRIABLE_4XX = frozenset({408, 425, 429})

    def read(self, size: int = -1) -> bytes:
        if size <= 0:
            size = _STREAM_CHUNK
        retries = 0
        while True:
            try:
                if not self.stream:
                    # If _open_stream itself fails, the exception falls into
                    # the same except block below and counts against the
                    # retry budget — no separate, swallowed reopen path.
                    self._open_stream()
                stream = self.stream
                if stream is None:
                    raise RuntimeError("HDFS stream open completed without a response")
                chunk = stream.read(size)
                if chunk:
                    self.bytes_read += len(chunk)
                return chunk
            except (
                requests.exceptions.RequestException,
                # urllib3 wraps ConnectionResetError / IncompleteRead / chunked
                # decoding failures into urllib3.exceptions.ProtocolError, which
                # is NOT an OSError nor a requests.RequestException. Without
                # this branch the retry/resume logic below is silently bypassed
                # and the WHOLE Ray Data pipeline aborts on the first transient
                # 'Connection reset by peer' from the WebHDFS DataNode.
                urllib3.exceptions.HTTPError,
                ConnectionError,
                TimeoutError,
                OSError,
            ) as e:
                # Fast-fail on deterministic client errors (auth/missing path/
                # bad request). Without this guard a 401 Unauthorized burns
                # the whole retry budget and reports the wrong error class to
                # the operator.
                if isinstance(e, requests.exceptions.HTTPError):
                    resp = getattr(e, "response", None)
                    status = (
                        getattr(resp, "status_code", None) if resp is not None else None
                    )
                    if (
                        status is not None
                        and 400 <= status < 500
                        and status not in self._RETRIABLE_4XX
                    ):
                        logger.error(
                            "Non-retriable HTTP %d for HDFS stream %s at offset %d",
                            status,
                            self.hdfs_file,
                            self.bytes_read,
                        )
                        raise RuntimeError(
                            f"HDFS stream rejected with HTTP {status}: {self.hdfs_file}"
                        ) from None

                retries += 1
                if retries > self.max_retries:
                    logger.error(
                        "Max retries exceeded for HDFS stream %s at offset %d (%s)",
                        self.hdfs_file,
                        self.bytes_read,
                        type(e).__name__,
                    )
                    raise RuntimeError(
                        f"HDFS stream failed after {self.max_retries} retries: "
                        f"{self.hdfs_file} ({type(e).__name__})"
                    ) from None
                logger.warning(
                    "Network error during HDFS stream read for %s at offset %d (%s). Retrying (%d/%d)...",
                    self.hdfs_file,
                    self.bytes_read,
                    type(e).__name__,
                    retries,
                    self.max_retries,
                )
                # Drop the broken response immediately so we do not leak the
                # underlying socket while we sleep before retrying.
                self._close_response(
                    f"Discarding broken stream after {type(e).__name__}"
                )
                time.sleep(min(2**retries, 60) + random.uniform(0, 1))
                # Intentionally do NOT reopen here. The next loop iteration
                # enters the try block and calls _open_stream() under the
                # same except guard, so an open failure also burns one
                # retry slot instead of being silently logged.

    def close(self):
        self._close_response("Explicit close requested")
        super().close()


def collect_hdfs_file_paths(hdfs_root: str, start: date, end: date) -> list[str]:
    """
    Enumerate all part-*.zst files on HDFS for the given date range.
    """
    return list(iter_hdfs_file_paths(hdfs_root, start, end))


def iter_hdfs_file_paths(
    hdfs_root: str,
    start: date,
    end: date,
    *,
    list_directory: Callable[[str], list[str]] | None = None,
) -> Iterator[str]:
    """Stream HDFS file discovery without retaining the complete manifest."""
    d = start

    # Local filesystem source
    if list_directory is None and os.path.exists(hdfs_root):
        while d <= end:
            date_dir = os.path.join(hdfs_root, f"ds={d.isoformat()}")
            if os.path.exists(date_dir):
                for f in sorted(
                    itertools.chain(
                        glob.iglob(os.path.join(date_dir, "part-*.zst")),
                        glob.iglob(os.path.join(date_dir, "hour=*", "part-*.zst")),
                    )
                ):
                    yield f
            d += timedelta(days=1)
        return

    list_directory = hdfs_ls if list_directory is None else list_directory
    while d <= end:
        date_dir = f"{hdfs_root}/ds={d.isoformat()}"
        entries = sorted(list_directory(date_dir))
        yield from (
            f
            for f in entries
            if f.rsplit("/", 1)[-1].startswith("part-") and f.endswith(".zst")
        )
        hour_dirs = [f for f in entries if f.rsplit("/", 1)[-1].startswith("hour=")]
        for f in hdfs_ls_parallel_iter(
            hour_dirs, max_workers=24, list_directory=list_directory
        ):
            fname = f.rsplit("/", 1)[-1]
            if fname != "_SUCCESS" and fname.endswith(".zst"):
                yield f
        d += timedelta(days=1)
