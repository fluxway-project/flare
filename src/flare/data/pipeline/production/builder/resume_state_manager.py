"""
flare.data.pipeline.production.builder.resume_state_manager — Atomic resume state for pipeline batches.

Three-state machine per HDFS file:
    pending → in_flight → processed | partial_processed | quarantined

State file is written atomically (tempfile + os.replace) so a crash mid-write
never corrupts the persisted state.

Block-level quarantine (5c.5b):
    When --block-quarantine-enabled, individual HDFS blocks that repeatedly fail
    are tracked in quarantined_blocks. SourceActor can skip known-bad blocks
    instead of quarantining the entire file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
from typing import Any

from pyarrow import fs

from flare.data.pipeline.production.builder.hdfs_source import HdfsPartitionCursor
from flare.data.pipeline.production.publication.contract import (
    PublicationLease,
    PublicationProtocol,
    validate_publication_id,
)

logger = logging.getLogger(__name__)

_STATE_VERSION = 1
_HDFS_BLOCK_SIZE = 128 * 1024 * 1024  # 128 MiB


def _block_id(hdfs_path: str, byte_offset: int) -> tuple[str, int]:
    """Quantize byte offset to HDFS block boundary."""
    return (hdfs_path, byte_offset // _HDFS_BLOCK_SIZE * _HDFS_BLOCK_SIZE)


class ResumeStateManager:
    """Atomic, crash-safe resume state manager for pipeline batches.

    All mutations auto-save to disk via tempfile + os.replace (atomic on
    POSIX). No explicit save() needed after mutation calls.

    Args:
        state_path: Local filesystem path for the JSON state file.
        s3_opts: Optional S3 storage options (reserved for future remote state).
    """

    def __init__(self, state_path: str, s3_opts: dict | None = None):
        self._path = state_path
        self._s3_opts = s3_opts or {}

        # Core state
        self._processed: set[str] = set()
        self._in_flight: dict[str, PublicationLease] = {}
        self._partial_processed: dict[str, dict] = {}
        self._quarantined: set[str] = set()
        self._attempts: dict[str, int] = {}
        self._partition_cursors: dict[str, HdfsPartitionCursor] = {}

        # Block-level quarantine (task 5c.5b)
        self._quarantined_blocks: set[tuple[str, int]] = set()
        self._block_attempts: dict[tuple[str, int], int] = {}

        self._lock = threading.RLock()

    def _get_fs_and_path(self) -> tuple:
        parsed = urllib.parse.urlparse(self._path)
        if parsed.scheme == "s3":
            s3_kwargs: dict[str, Any] = {"background_writes": False}
            if "endpoint" in self._s3_opts:
                s3_kwargs["endpoint_override"] = self._s3_opts["endpoint"]
                s3_kwargs["force_virtual_addressing"] = False
                if self._s3_opts["endpoint"].startswith("http://"):
                    s3_kwargs["scheme"] = "http"
            if "access_key" in self._s3_opts:
                s3_kwargs["access_key"] = self._s3_opts["access_key"]
            if "secret_key" in self._s3_opts:
                s3_kwargs["secret_key"] = self._s3_opts["secret_key"]
            if "region" in self._s3_opts:
                s3_kwargs["region"] = self._s3_opts["region"]
            fs_inst = fs.S3FileSystem(**s3_kwargs)
            return fs_inst, parsed.netloc + parsed.path
        return fs.FileSystem.from_uri(self._path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def processed(self) -> set[str]:
        return self._processed

    @property
    def partial_processed(self) -> dict[str, dict]:
        return self._partial_processed

    @property
    def quarantined(self) -> set[str]:
        return self._quarantined

    @property
    def quarantined_blocks(self) -> set[tuple[str, int]]:
        return self._quarantined_blocks

    @property
    def all_done(self) -> set[str]:
        """All files that should not be reprocessed (processed + partial)."""
        return self._processed | set(self._partial_processed.keys())

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> tuple[set[str], dict[str, PublicationLease]]:
        """Load state from disk. Returns (processed, in_flight).

        Rejects any version or field set outside the current protocol.
        """
        with self._lock:
            parsed = urllib.parse.urlparse(self._path)

            try:
                if parsed.scheme == "s3":
                    import boto3
                    from botocore.exceptions import ClientError

                    client = boto3.client(
                        "s3",
                        endpoint_url=self._s3_opts.get("endpoint"),
                        aws_access_key_id=self._s3_opts.get("access_key"),
                        aws_secret_access_key=self._s3_opts.get("secret_key"),
                        region_name=self._s3_opts.get("region"),
                    )
                    try:
                        resp = client.get_object(
                            Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
                        )
                        data = json.loads(resp["Body"].read().decode("utf-8"))
                    except ClientError as e:
                        if e.response["Error"]["Code"] == "NoSuchKey":
                            return set(), {}
                        raise
                else:
                    fs_inst, fs_path = self._get_fs_and_path()
                    with fs_inst.open_input_stream(fs_path) as f:
                        data = json.loads(f.read().decode("utf-8"))
            except FileNotFoundError:
                return set(), {}

            version = int(data["version"])
            if version != _STATE_VERSION:
                raise ValueError(
                    f"resume state version must be {_STATE_VERSION}, got {version}"
                )
            expected_fields = {
                "version",
                "processed",
                "in_flight",
                "attempts",
                "quarantined",
                "partial_processed",
                "quarantined_blocks",
                "block_attempts",
                "partition_cursors",
            }
            if set(data) != expected_fields:
                raise ValueError(
                    "resume state fields do not match protocol: "
                    f"expected={sorted(expected_fields)} actual={sorted(data)}"
                )
            self._processed = set(data["processed"])
            if not isinstance(data["in_flight"], dict):
                raise TypeError("resume state in_flight must be an object")
            self._in_flight = {
                validate_publication_id(publication_id): PublicationLease.from_state(
                    lease
                )
                for publication_id, lease in data["in_flight"].items()
            }
            self._attempts = data["attempts"]
            self._quarantined = set(data["quarantined"])
            self._partial_processed = data["partial_processed"]
            self._quarantined_blocks = {tuple(b) for b in data["quarantined_blocks"]}
            self._block_attempts = {
                tuple(json.loads(k)): v for k, v in data["block_attempts"].items()
            }
            self._partition_cursors = {
                partition: HdfsPartitionCursor.from_checkpoint(cursor_state)
                for partition, cursor_state in data["partition_cursors"].items()
            }

            # The manager owns the processed index; returning it avoids a
            # second full-size set on the driver for large campaigns.
            return self._processed, dict(self._in_flight)

    def save(self) -> None:
        """Atomically persist state to disk (tempfile + os.replace)."""
        with self._lock:
            data = {
                "version": _STATE_VERSION,
                "processed": sorted(self._processed),
                "in_flight": {
                    publication_id: lease.to_state()
                    for publication_id, lease in sorted(self._in_flight.items())
                },
                "attempts": self._attempts,
                "quarantined": sorted(self._quarantined),
                "partial_processed": self._partial_processed,
                "quarantined_blocks": [
                    list(b) for b in sorted(self._quarantined_blocks)
                ],
                "block_attempts": {
                    json.dumps(list(k)): v for k, v in self._block_attempts.items()
                },
                "partition_cursors": {
                    partition: cursor.checkpoint_state()
                    for partition, cursor in sorted(self._partition_cursors.items())
                },
            }

            import tempfile

            # Write atomically using a local tempfile first, then uploading to target via pyarrow fs
            fd, tmp_path = tempfile.mkstemp(prefix=".resume_state_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)

                parsed = urllib.parse.urlparse(self._path)
                if parsed.scheme == "s3":
                    import boto3

                    client = boto3.client(
                        "s3",
                        endpoint_url=self._s3_opts.get("endpoint"),
                        aws_access_key_id=self._s3_opts.get("access_key"),
                        aws_secret_access_key=self._s3_opts.get("secret_key"),
                        region_name=self._s3_opts.get("region"),
                    )
                    with open(tmp_path, "rb") as lf:
                        client.put_object(
                            Bucket=parsed.netloc,
                            Key=parsed.path.lstrip("/"),
                            Body=lf.read(),
                        )
                else:
                    fs_inst, fs_path = self._get_fs_and_path()
                    # Atomically copy file to remote fs (or local)
                    fs_tmp_path = f"{fs_path}.tmp.{os.getpid()}"
                    fs_inst.create_dir(os.path.dirname(fs_path), recursive=True)
                    with (
                        open(tmp_path, "rb") as lf,
                        fs_inst.open_output_stream(fs_tmp_path) as rf,
                    ):
                        rf.write(lf.read())
                    fs_inst.move(fs_tmp_path, fs_path)
            finally:
                os.unlink(tmp_path)

    def clear(self) -> None:
        """Reset all state."""
        with self._lock:
            self._processed.clear()
            self._in_flight.clear()
            self._partial_processed.clear()
            self._quarantined.clear()
            self._attempts.clear()
            self._quarantined_blocks.clear()
            self._block_attempts.clear()
            self._partition_cursors.clear()
            self.save()

    # ------------------------------------------------------------------
    # Batch lifecycle (three-phase commit)
    # ------------------------------------------------------------------

    def mark_in_flight(
        self,
        publication_id: str,
        files: list[str],
        protocol: PublicationProtocol,
    ) -> None:
        """Record intent before processing (in-flight marker)."""
        with self._lock:
            validate_publication_id(publication_id)
            if publication_id in self._in_flight:
                raise ValueError(f"publication is already in flight: {publication_id}")
            self._in_flight[publication_id] = PublicationLease(
                files=tuple(files),
                protocol=protocol,
            )
            self.save()

    def complete_in_flight(
        self, publication_id: str, partial_meta: dict | None = None
    ) -> None:
        """Mark batch complete after vortex_commit.

        Files listed in partial_meta go to partial_processed (not processed).
        All other files in the batch go to processed.
        """
        with self._lock:
            lease = self._in_flight.pop(publication_id, None)
            if lease is None:
                raise KeyError(f"publication is not in flight: {publication_id}")

            for f in lease.files:
                if partial_meta is not None and f in partial_meta:
                    self._partial_processed[f] = partial_meta[f]
                else:
                    self._processed.add(f)

            self.save()

    def discard_in_flight(self, publication_id: str) -> None:
        """Discard an in-flight batch (failed, will retry)."""
        with self._lock:
            if self._in_flight.pop(publication_id, None) is None:
                raise KeyError(f"publication is not in flight: {publication_id}")
            self.save()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def filter_pending(self, all_files: list[str]) -> list[str]:
        """Return files not yet processed/partial/quarantined."""
        with self._lock:
            done = self.all_done | self._quarantined
            return [f for f in all_files if f not in done]

    def is_pending(self, file_path: str) -> bool:
        """Check one discovered file without materializing a discovery list."""
        with self._lock:
            return (
                file_path not in self._processed
                and file_path not in self._partial_processed
                and file_path not in self._quarantined
                and all(
                    file_path not in lease.files for lease in self._in_flight.values()
                )
            )

    def partition_cursor(self, partition: str) -> HdfsPartitionCursor | None:
        with self._lock:
            return self._partition_cursors.get(partition)

    def update_partition_cursor(self, cursor: HdfsPartitionCursor) -> None:
        with self._lock:
            if self._partition_cursors.get(cursor.partition) == cursor:
                return
            self._partition_cursors[cursor.partition] = cursor
            self.save()

    def add_processed(self, files: list[str]) -> None:
        """Directly mark files as processed."""
        with self._lock:
            self._processed.update(files)
            self.save()

    # ------------------------------------------------------------------
    # Fault tolerance: file-level
    # ------------------------------------------------------------------

    def record_failure(self, files: list[str], max_attempts: int) -> list[str]:
        """Increment attempt counter. Quarantine files exceeding threshold.

        Returns list of newly quarantined file paths.
        """
        with self._lock:
            newly_quarantined = []
            for f in files:
                self._attempts[f] = self._attempts.get(f, 0) + 1
                if self._attempts[f] >= max_attempts:
                    self._quarantined.add(f)
                    newly_quarantined.append(f)
            self.save()
            return newly_quarantined

    def reset_attempts(self, files: list[str]) -> None:
        """Reset attempt counters for successfully processed files."""
        with self._lock:
            for f in files:
                self._attempts.pop(f, None)
            # No save here — caller typically saves via complete_in_flight

    def add_quarantined(self, files: list[str]) -> list[str]:
        """Manually quarantine files. Returns newly added paths."""
        with self._lock:
            newly = [f for f in files if f not in self._quarantined]
            self._quarantined.update(newly)
            if newly:
                self.save()
            return newly

    # ------------------------------------------------------------------
    # Partial recovery
    # ------------------------------------------------------------------

    def retry_partial(self, files: list[str] | None = None) -> list[str]:
        """Move partial_processed files back to pending for reprocessing.

        Args:
            files: Specific files to retry (None = all partial).

        Returns:
            List of files moved back to pending.
        """
        with self._lock:
            if files is None:
                retried = list(self._partial_processed.keys())
                self._partial_processed.clear()
            else:
                retried = [f for f in files if f in self._partial_processed]
                for f in retried:
                    del self._partial_processed[f]

            if retried:
                self.save()
            return retried

    # ------------------------------------------------------------------
    # Block-level quarantine (task 5c.5b)
    # ------------------------------------------------------------------

    def record_block_failure(
        self, hdfs_path: str, byte_offset: int, max_attempts: int
    ) -> int:
        """Record a block-level read failure.

        Args:
            hdfs_path: HDFS file path.
            byte_offset: Byte offset where failure occurred.
            max_attempts: Threshold before quarantining the block.

        Returns:
            Current attempt count for this block.
        """
        with self._lock:
            bid = _block_id(hdfs_path, byte_offset)
            self._block_attempts[bid] = self._block_attempts.get(bid, 0) + 1
            count = self._block_attempts[bid]

            if count >= max_attempts:
                self._quarantined_blocks.add(bid)
                logger.warning(
                    "Block quarantined: %s offset=%d (attempts=%d)",
                    hdfs_path,
                    bid[1],
                    count,
                )

            self.save()
            return count

    def is_block_quarantined(self, hdfs_path: str, byte_offset: int) -> bool:
        """Check if a specific block is quarantined."""
        return _block_id(hdfs_path, byte_offset) in self._quarantined_blocks

    def quarantined_block_ratio(self, hdfs_path: str) -> float:
        """Fraction of blocks quarantined for a given file (0.0 - 1.0).

        Used to decide when to escalate to file-level quarantine.
        """
        path_blocks = [b for b in self._quarantined_blocks if b[0] == hdfs_path]
        if not path_blocks:
            return 0.0
        # Estimate total blocks from max quarantined offset
        max_offset = max(b[1] for b in path_blocks)
        estimated_total = max_offset // _HDFS_BLOCK_SIZE + 2  # +2 for safety
        return len(path_blocks) / estimated_total
