"""Process-level exclusion for publishers sharing a POSIX resume file."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def publication_run_lock(state_path: str):
    """Hold an advisory lock across reconciliation, publication and checkpointing.

    The lock file must be on the same shared POSIX filesystem as the state.
    It is intentionally never unlinked: unlinking a locked inode allows a
    second publisher to acquire a different inode at the same pathname.
    """
    if "://" in state_path:
        raise ValueError("publication_run_lock requires a shared POSIX state path")
    path = Path(state_path + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another producer holds the resume lock: {path}"
            ) from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
