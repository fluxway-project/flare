from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

from flare.data.pipeline.consumption.source.batching.contracts import (
    AssembledBatch,
    BatchCompletionPolicy,
    CompletionContext,
)


class CompletionPolicy(str, Enum):
    CARRY = "carry"
    DROP = "drop"
    PAD = "pad"
    REQUIRE = "require"

    @classmethod
    def from_value(cls, value: str | CompletionPolicy) -> CompletionPolicy:
        if isinstance(value, cls):
            return value
        v = value.lower()
        if v in ("drop", "drop_remainder"):
            return cls.DROP
        if v in ("require", "exact_aligned"):
            return cls.REQUIRE
        if v in ("carry",):
            return cls.CARRY
        if v in ("pad",):
            return cls.PAD
        raise ValueError(f"Unknown completion policy {value!r}")


PolicyFactory = Callable[[], BatchCompletionPolicy]


class CarryUntilComplete:
    def finalize(
        self, context: CompletionContext, *, terminal: bool
    ) -> tuple[AssembledBatch[object], ...]:
        if terminal:
            context.discard_pending(count_as_discarded=True)
        return ()


class DropIncompleteAtEnd:
    def finalize(
        self, context: CompletionContext, *, terminal: bool
    ) -> tuple[AssembledBatch[object], ...]:
        if terminal:
            context.discard_pending(count_as_discarded=True)
        return ()


class PadWithValidity:
    def finalize(
        self, context: CompletionContext, *, terminal: bool
    ) -> tuple[AssembledBatch[object], ...]:
        if not terminal or context.pending_rows == 0:
            return ()
        return (context.pad_pending(),)


class RequireComplete:
    def finalize(
        self, context: CompletionContext, *, terminal: bool
    ) -> tuple[AssembledBatch[object], ...]:
        if terminal and context.pending_rows:
            raise ValueError(
                "incomplete batch encountered under RequireComplete: "
                f"rows={context.pending_rows} batch_size={context.batch_size}"
            )
        return ()


class CompletionPolicyRegistry:
    def __init__(self, factories: Mapping[str, PolicyFactory]) -> None:
        if not factories:
            raise ValueError("completion policy registry must not be empty")
        validated: dict[str, PolicyFactory] = {}
        for name, factory in factories.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ValueError("completion policy names must be canonical strings")
            if not callable(factory):
                raise TypeError(f"completion policy factory is not callable: {name}")
            validated[name] = factory
        self._factories = MappingProxyType(validated)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def create(self, name: Any) -> BatchCompletionPolicy:
        try:
            key = CompletionPolicy.from_value(name).value
        except ValueError:
            key = str(getattr(name, "value", name))
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(
                f"unknown completion policy {name!r}; registered={self.names!r}"
            )
        policy = factory()
        if not isinstance(policy, BatchCompletionPolicy):
            raise TypeError(f"completion policy factory returned invalid value: {name}")
        return policy

    def extend(
        self, factories: Mapping[str, PolicyFactory]
    ) -> CompletionPolicyRegistry:
        overlap = sorted(set(self._factories) & set(factories))
        if overlap:
            raise ValueError(f"completion policies already registered: {overlap}")
        return CompletionPolicyRegistry({**self._factories, **factories})


STANDARD_COMPLETION_POLICIES = CompletionPolicyRegistry(
    {
        "carry": CarryUntilComplete,
        "drop": DropIncompleteAtEnd,
        "pad": PadWithValidity,
        "require": RequireComplete,
    }
)
COMPLETION_POLICY_NAMES = STANDARD_COMPLETION_POLICIES.names


def validate_completion_policy(policy: str) -> str:
    STANDARD_COMPLETION_POLICIES.create(policy)
    return policy


def completion_policy_for_name(name: str) -> BatchCompletionPolicy:
    return STANDARD_COMPLETION_POLICIES.create(name)
