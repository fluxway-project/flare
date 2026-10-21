from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)


class ExecutionTracer(ABC):
    @abstractmethod
    def record_event(
        self,
        stage: str,
        *,
        step: int | None,
        spmd_id: str,
        rank: int,
        group: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class _AsyncJsonlWriter:
    def __init__(self, root_dir: str, *, capacity: int = 8192) -> None:
        self.root_dir = root_dir
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(capacity)
        self._failure: BaseException | None = None
        self._closed = False
        os.makedirs(root_dir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="flare-execution-trace-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, record: dict[str, Any]) -> None:
        self._raise_failure()
        if self._closed:
            raise RuntimeError("execution trace writer is closed")
        try:
            self._queue.put_nowait(record)
        except queue.Full as error:
            raise RuntimeError(
                "execution trace writer queue is full; diagnostic capture cannot keep up"
            ) from error

    def flush(self) -> None:
        self._queue.join()
        self._raise_failure()

    def close(self) -> None:
        if self._thread.is_alive():
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self._raise_failure()

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("execution trace writer failed") from self._failure

    def _run(self) -> None:
        stop = False
        try:
            while not stop:
                first = self._queue.get()
                if first is None:
                    self._queue.task_done()
                    break
                batch = [first]
                while len(batch) < 256:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        self._queue.task_done()
                        stop = True
                        break
                    batch.append(item)
                try:
                    self._write_batch(batch)
                finally:
                    for _ in batch:
                        self._queue.task_done()
        except BaseException as error:
            self._failure = error
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[bytes]] = {}
        for record in batch:
            filename = (
                f"{record['stage']}."
                f"{sha256(record['spmd_id'].encode('utf-8')).hexdigest()}."
                f"{record['rank']}.jsonl"
            )
            encoded = (
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
            grouped.setdefault(os.path.join(self.root_dir, filename), []).append(
                encoded
            )
        for path, records in grouped.items():
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                encoded = b"".join(records)
                written = os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
            if written != len(encoded):
                raise OSError(
                    f"short execution trace write: expected={len(encoded)} actual={written}"
                )


_WRITER_LOCK = threading.Lock()
_WRITERS: dict[tuple[int, str], _AsyncJsonlWriter] = {}


def _writer_for(root_dir: str) -> _AsyncJsonlWriter:
    key = (os.getpid(), root_dir)
    with _WRITER_LOCK:
        writer = _WRITERS.get(key)
        if writer is None:
            writer = _AsyncJsonlWriter(root_dir)
            _WRITERS[key] = writer
        return writer


def _close_trace_writers() -> None:
    pid = os.getpid()
    with _WRITER_LOCK:
        writers = tuple(
            writer for (writer_pid, _), writer in _WRITERS.items() if writer_pid == pid
        )
    for writer in writers:
        try:
            writer.close()
        except BaseException:
            _LOGGER.exception("execution trace writer shutdown failed")


atexit.register(_close_trace_writers)


@dataclass(frozen=True, slots=True)
class ExecutionTraceOptions:
    directory: str | None = None
    include_values: bool = False

    def __post_init__(self) -> None:
        if self.directory is not None and not self.directory:
            raise ValueError("execution trace directory must not be empty")
        if not isinstance(self.include_values, bool):
            raise TypeError("execution trace include_values must be bool")

    @property
    def enabled(self) -> bool:
        return self.directory is not None


@dataclass(frozen=True)
class JsonlExecutionTracer(ExecutionTracer):
    root_dir: str
    include_values: bool = False

    def __post_init__(self) -> None:
        if not self.root_dir:
            raise ValueError("execution trace root_dir must not be empty")

    def prepare(self) -> None:
        _writer_for(self.root_dir)

    def record_event(
        self,
        stage: str,
        *,
        step: int | None,
        spmd_id: str,
        rank: int,
        group: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(spmd_id, str) or not spmd_id:
            raise ValueError(
                f"execution trace event {stage} requires a non-empty spmd_id"
            )
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(
                f"execution trace event {stage} requires a non-negative rank"
            )
        record = {
            "stage": stage,
            "group": group,
            "step": step,
            "spmd_id": spmd_id,
            "rank": rank,
            "payload": payload or {},
        }
        _writer_for(self.root_dir).submit(record)

    def flush(self) -> None:
        _writer_for(self.root_dir).flush()

    def close(self) -> None:
        _writer_for(self.root_dir).close()


def create_execution_tracer(
    options: ExecutionTraceOptions,
) -> ExecutionTracer | None:
    if not isinstance(options, ExecutionTraceOptions):
        raise TypeError("options must be ExecutionTraceOptions")
    if not options.enabled:
        return None
    if options.directory is None:
        raise RuntimeError("enabled execution trace options require a directory")
    return JsonlExecutionTracer(
        os.path.abspath(options.directory), options.include_values
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _numeric_summary(value: Any) -> dict[str, Any]:
    array = _to_numpy(value)
    summary: dict[str, Any] = {
        "shape": [int(dimension) for dimension in array.shape],
        "dtype": str(array.dtype),
        "size": array.size,
    }
    if array.size:
        if np.issubdtype(array.dtype, np.floating):
            finite_mask = np.isfinite(array)
            summary.update(
                finite_count=int(np.count_nonzero(finite_mask)),
                nan_count=int(np.count_nonzero(np.isnan(array))),
                inf_count=int(np.count_nonzero(np.isinf(array))),
            )
            finite = array[finite_mask]
        else:
            summary.update(finite_count=array.size, nan_count=0, inf_count=0)
            finite = array
        if finite.size:
            summary.update(
                min=_scalar(finite.min()),
                max=_scalar(finite.max()),
                sum=_scalar(finite.sum()),
            )
    else:
        summary.update(finite_count=0, nan_count=0, inf_count=0)
    return summary


def _descriptor_summary(value: Any) -> dict[str, Any]:
    flat_values = getattr(value, "flat_values", value)
    global_shape = getattr(value, "global_shape", None)
    origin_indices = getattr(value, "origin_indices", None)
    nested_splits = getattr(value, "nested_splits", ()) or ()
    shape = tuple(getattr(flat_values, "shape", ()))
    return {
        "flat_shape": [int(dimension) for dimension in shape],
        "flat_dtype": str(getattr(flat_values, "dtype", "unknown")),
        "flat_size": int(prod(shape)),
        "nested_levels": len(nested_splits),
        "global_shape": None
        if global_shape is None
        else [int(dimension) for dimension in global_shape],
        "origin_type": None
        if origin_indices is None
        else type(origin_indices).__name__,
    }


def _flat_int_values(value: Any) -> list[int]:
    return [int(item) for item in _to_numpy(value).reshape(-1)]


def _values(value: Any) -> list[Any]:
    return _to_numpy(value).tolist()


def _leading_row_count(value: Any) -> int:
    shape = tuple(getattr(value, "shape", ()))
    return int(prod(shape) if not shape else shape[0])


def _metadata_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"shape": [], "dtype": "none", "size": 0}
    shape = tuple(getattr(value, "shape", ()))
    return {
        "shape": [int(dimension) for dimension in shape],
        "dtype": str(getattr(value, "dtype", "unknown")),
        "size": int(prod(shape)),
    }


def _scalar(value: Any) -> int | float:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, np.integer):
        return int(scalar)
    if isinstance(scalar, np.floating):
        return float(scalar)
    return scalar


__all__ = [
    "ExecutionTraceOptions",
    "ExecutionTracer",
    "JsonlExecutionTracer",
    "create_execution_tracer",
]
