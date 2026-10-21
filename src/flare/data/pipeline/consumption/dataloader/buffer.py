"""Explicit CPU buffer ownership for dataloader payloads."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class BufferCapabilities:
    immutable: bool
    leased: bool
    device_transfer: bool = False


class BufferBackend:
    """Minimal lifecycle boundary between decoded data and a consumer."""

    capabilities: BufferCapabilities

    def view(self) -> Any:
        raise NotImplementedError

    @property
    def device(self) -> str:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def dtype(self) -> np.dtype[Any]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class ImmutableCpuBuffer(BufferBackend):
    capabilities = BufferCapabilities(immutable=True, leased=False)

    def __init__(self, value: Any) -> None:
        array = np.asarray(value)
        array.setflags(write=False)
        self._array = array
        self._released = False

    def view(self) -> np.ndarray:
        if self._released:
            raise RuntimeError("immutable buffer was released")
        return self._array

    @property
    def device(self) -> Literal["cpu"]:
        return "cpu"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    def release(self) -> None:
        self._released = True


class LeasedCpuBuffer(BufferBackend):
    capabilities = BufferCapabilities(immutable=False, leased=True)

    def __init__(
        self, value: Any, *, releaser: Callable[[np.ndarray], None] | None = None
    ) -> None:
        self._array = np.asarray(value)
        self._releaser = releaser
        self._released = False
        self._lock = threading.Lock()
        self._references = 1

    def retain(self) -> LeasedCpuBuffer:
        with self._lock:
            if self._released:
                raise RuntimeError("cannot retain a released leased buffer")
            self._references += 1
        return self

    def view(self) -> np.ndarray:
        with self._lock:
            if self._released:
                raise RuntimeError("leased buffer was released")
            return self._array

    @property
    def device(self) -> Literal["cpu"]:
        return "cpu"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    def release(self) -> None:
        releaser = None
        with self._lock:
            if self._released:
                raise RuntimeError("leased buffer released more than once")
            self._references -= 1
            if self._references == 0:
                self._released = True
                releaser = self._releaser
        if releaser is not None:
            releaser(self._array)
