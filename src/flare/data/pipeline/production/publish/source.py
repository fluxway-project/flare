from __future__ import annotations

from contextlib import AbstractContextManager
from typing import BinaryIO, Protocol


class PartitionedSourceSession(Protocol):
    @property
    def root(self) -> str: ...

    def list_directory(self, path: str) -> list[str]: ...

    def open(self, path: str) -> BinaryIO: ...


class PartitionedSourceProvider(Protocol):
    def open(self) -> AbstractContextManager[PartitionedSourceSession]: ...
