from __future__ import annotations

from flare.kernels.backends.cupy_memory import (
    EagerCupyPoolReclaimer,
    RetainCupyPoolReclaimer,
)
from flare.kernels.backends.gpu_cupy import CupySparseEmbeddingOperator


class _Pool:
    def __init__(self, cached: int = 0) -> None:
        self.cached = cached
        self.releases = 0

    def free_bytes(self) -> int:
        return self.cached

    def free_all_blocks(self) -> None:
        self.cached = 0
        self.releases += 1


class _Cupy:
    def __init__(self, *, cached: int) -> None:
        self.device = _Pool(cached)
        self.pinned = _Pool()

    def get_default_memory_pool(self) -> _Pool:
        return self.device

    def get_default_pinned_memory_pool(self) -> _Pool:
        return self.pinned


def test_pool_policies_preserve_eager_and_retained_modes() -> None:
    cupy = _Cupy(cached=200)
    EagerCupyPoolReclaimer().after_operation(cupy)
    assert (cupy.device.releases, cupy.pinned.releases) == (1, 1)
    RetainCupyPoolReclaimer().after_operation(cupy)
    assert (cupy.device.releases, cupy.pinned.releases) == (1, 1)


def test_operator_delegates_pool_lifecycle_to_injected_policy() -> None:
    events: list[str] = []

    class _Stream:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    class _Policy:
        def after_operation(self, _cupy) -> None:
            events.append("reclaim")

    operator = object.__new__(CupySparseEmbeddingOperator)
    operator.stream = _Stream()
    operator.cp = _Cupy(cached=200)
    operator._memory_reclaimer = _Policy()
    with operator.execution_context():
        events.append("work")
    assert events == ["enter", "work", "reclaim", "exit"]


def test_cupy_operator_maps_legacy_flag_and_accepts_policy_injection() -> None:
    eager = CupySparseEmbeddingOperator("float16", "float32", empty_cache=True)
    retained = CupySparseEmbeddingOperator("float16", "float32", empty_cache=False)
    injected = RetainCupyPoolReclaimer()
    custom = CupySparseEmbeddingOperator(
        "float16", "float32", memory_reclaimer=injected
    )
    assert isinstance(eager._memory_reclaimer, EagerCupyPoolReclaimer)
    assert isinstance(retained._memory_reclaimer, RetainCupyPoolReclaimer)
    assert custom._memory_reclaimer is injected
