from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any


class SourceCursor(Mapping[str, Any]):
    __slots__ = ("_payload",)

    def __init__(self, value: Mapping[str, Any] | SourceCursor) -> None:
        if isinstance(value, SourceCursor):
            self._payload = value._payload
            return
        if not isinstance(value, Mapping):
            raise TypeError("source cursor must be a mapping")
        self._payload = MappingProxyType(copy.deepcopy(dict(value)))

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SourceCursor):
            return self._payload == other._payload
        if isinstance(other, Mapping):
            return dict(self._payload) == dict(other)
        return False

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._payload.items())))

    def __deepcopy__(self, memo: dict[int, Any]) -> SourceCursor:
        return self

    def __repr__(self) -> str:
        return f"SourceCursor({dict(self._payload)!r})"

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._payload))

    def __reduce__(self) -> tuple[Any, ...]:
        return (self.__class__, (dict(self._payload),))


__all__ = ["SourceCursor"]
