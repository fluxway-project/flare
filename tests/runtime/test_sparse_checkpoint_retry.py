import asyncio

import pyarrow as pa
import pytest

from flare.runtime.fluxway.checkpoint.sparse import (
    SparseEmbeddingCheckpointComponent,
    _RetryingSparseCheckpointComponent,
)


@pytest.mark.parametrize("method", ["publish_save", "acknowledge_save"])
def test_transient_lance_read_is_retried(monkeypatch, method):
    calls = 0

    async def fail_twice(_self, _argument):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise pa.ArrowInvalid("transient manifest read")

    monkeypatch.setattr(SparseEmbeddingCheckpointComponent, method, fail_twice)
    component = object.__new__(_RetryingSparseCheckpointComponent)
    asyncio.run(getattr(component, method)(object()))
    assert calls == 3


def test_persistent_lance_read_error_is_not_hidden(monkeypatch):
    calls = 0

    async def always_fail(_self, _argument):
        nonlocal calls
        calls += 1
        raise pa.ArrowInvalid("persistent corruption")

    monkeypatch.setattr(SparseEmbeddingCheckpointComponent, "publish_save", always_fail)
    component = object.__new__(_RetryingSparseCheckpointComponent)
    with pytest.raises(pa.ArrowInvalid, match="persistent corruption"):
        asyncio.run(component.publish_save(object()))
    assert calls == 3
