from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Generic, TypeVar

T = TypeVar("T")
Factory = Callable[[Mapping[str, Any]], T]


class FactoryRegistry(Generic[T]):
    def __init__(self, factories: Mapping[str, Factory[T]]) -> None:
        if not factories:
            raise ValueError("factory registry must not be empty")
        validated: dict[str, Factory[T]] = {}
        for kind, factory in factories.items():
            if not isinstance(kind, str) or not kind or kind.strip() != kind:
                raise ValueError("factory kind must be a canonical string")
            if not callable(factory):
                raise TypeError(f"factory is not callable: {kind}")
            validated[kind] = factory
        self._factories = MappingProxyType(validated)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def create(self, kind: str, options: Mapping[str, Any]) -> T:
        factory = self._factories.get(kind)
        if factory is None:
            raise ValueError(
                f"unknown factory kind {kind!r}; registered={self.kinds!r}"
            )
        return factory(options)

    def extend(self, factories: Mapping[str, Factory[T]]) -> FactoryRegistry[T]:
        overlap = sorted(set(self._factories) & set(factories))
        if overlap:
            raise ValueError(f"factory kinds already registered: {overlap}")
        return FactoryRegistry({**self._factories, **factories})
