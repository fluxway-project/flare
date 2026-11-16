"""CuPy pool-reclamation policies, independent of sparse kernel execution."""

from __future__ import annotations

from typing import Any, Protocol


class CupyPoolReclaimer(Protocol):
    def after_operation(self, cupy: Any) -> None: ...


class EagerCupyPoolReclaimer:
    """Preserve the original release-on-every-operation behavior."""

    def after_operation(self, cupy: Any) -> None:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


class RetainCupyPoolReclaimer:
    """Compatibility for callers disabling cleanup; default production pipelines do not use it."""

    def after_operation(self, cupy: Any) -> None:
        return None


__all__ = [
    "CupyPoolReclaimer",
    "EagerCupyPoolReclaimer",
    "RetainCupyPoolReclaimer",
]
