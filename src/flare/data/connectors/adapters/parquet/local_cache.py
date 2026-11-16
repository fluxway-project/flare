"""Bounded node-local cache for immutable Iceberg Parquet data files."""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO


class LocalParquetFileCache:
    def __init__(self, max_bytes: int, directory: Path | None = None) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("local Parquet cache size must be positive")
        self.max_bytes = max_bytes
        self.directory = directory or Path("/tmp") / (
            f"flare-parquet-cache-{os.getuid()}"
        )
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.stat().st_uid != os.getuid():
            raise PermissionError("local Parquet cache must belong to the current user")
        self.directory.chmod(0o700)

    def open_file(
        self,
        *,
        revision_identity: str,
        locator: str,
        input_file_factory: Callable[[], Any],
    ) -> BinaryIO:
        if not revision_identity or not locator:
            raise ValueError("local Parquet cache requires revision and locator")
        digest = hashlib.sha256(
            (revision_identity + "\0" + locator).encode("utf-8")
        ).hexdigest()
        target = self.directory / f"{digest}.parquet"
        lock_path = self.directory / f"{digest}.lock"
        downloaded = False
        size = 0
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.is_file() or target.stat().st_size == 0:
                for stale in self.directory.glob(f"{digest}.*.partial"):
                    stale.unlink()
                partial = self.directory / f"{digest}.{os.getpid()}.partial"
                try:
                    with (
                        input_file_factory().open() as remote,
                        partial.open("xb") as local,
                    ):
                        while chunk := remote.read(1024 * 1024):
                            local.write(chunk)
                    if partial.stat().st_size == 0:
                        raise ValueError("remote Parquet file is empty")
                    os.replace(partial, target)
                    downloaded = True
                finally:
                    partial.unlink(missing_ok=True)
            os.utime(target, None)
            size = target.stat().st_size
            handle = target.open("rb")
            fcntl.flock(lock, fcntl.LOCK_UN)
        if downloaded or size > self.max_bytes:
            self._evict(exclude=target if size <= self.max_bytes else None)
        return handle

    def _evict(self, *, exclude: Path | None) -> None:
        with (self.directory / ".evict.lock").open("a+b") as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX)
            files = sorted(
                self.directory.glob("*.parquet"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            total = sum(path.stat().st_size for path in files)
            for path in files:
                if total <= self.max_bytes:
                    break
                if path == exclude:
                    continue
                with path.with_suffix(".lock").open("a+b") as file_lock:
                    try:
                        fcntl.flock(file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if path.exists():
                        size = path.stat().st_size
                        path.unlink()
                        total -= size
                    fcntl.flock(file_lock, fcntl.LOCK_UN)
            if total > self.max_bytes and exclude is not None and exclude.exists():
                exclude.unlink()


__all__ = ["LocalParquetFileCache"]
