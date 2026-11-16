import asyncio
from contextlib import asynccontextmanager

import numpy as np
from sparse_embedding.checkpoint import PartitionSpec

from flare.runtime.fluxway.transport import make_transport_tensor
from flare.runtime.fluxway.workers import dynamic_embedding


def test_sparse_native_thread_budget_does_not_read_ray_cpu_allocation(monkeypatch):
    seen = []

    class Bound(dynamic_embedding.DynamicSparseEmbeddingSPMD):
        _factory_assignment_spec = PartitionSpec.modulo()
        _factory_resource_id = "test.assignment"
        _factory_thread_budget = 64

    class Store:
        def __init__(self, *, embedding_config, **_kwargs):
            seen.append(embedding_config.thread_budget)

    monkeypatch.setattr(
        dynamic_embedding, "register_sparse_partition_resource", lambda *_: None
    )
    monkeypatch.setattr(dynamic_embedding.se, "EmbeddingStore", Store)
    owner = object.__new__(Bound)
    Bound.__init__(
        owner,
        gname="test",
        spec={
            "emb_dims": [2],
            "value_dtype": "float32",
            "gradient_dtype": "float32",
            "gradient_transport_dtype": "float32",
            "optimizer_dtype": "float32",
            "grad_accum_steps": 1,
            "key_dtype": "int64",
            "optimizer_enabled": False,
            "sparse_reserve_capacity": 0,
        },
        LookupOutput=object,
    )
    assert seen == [64]


def test_sparse_accumulation_scope_preserves_dimension(monkeypatch):
    observed = []

    @asynccontextmanager
    async def scoped(name, **attributes):
        observed.append((name, attributes))
        yield

    class Store:
        async def async_accumulate_gradients(self, keys, gradient):
            observed.append(("native_accumulate", int(keys.size)))

    monkeypatch.setattr(dynamic_embedding.profiler, "enabled", lambda: True)
    monkeypatch.setattr(dynamic_embedding.profiler, "async_scope", scoped)
    owner = object.__new__(dynamic_embedding.DynamicSparseEmbeddingSPMD)
    owner.gname = "test_group"
    owner.gradient_transport_dtype = "float32"
    owner.gradient_dtype = "float32"
    owner.store = Store()
    keys = np.asarray([11, 12], dtype=np.int64)
    gradient = make_transport_tensor(np.ones((2, 4), dtype=np.float32))

    asyncio.run(owner._accumulate_gradients(0, {4: [(keys, gradient)]}))

    assert observed[0] == (
        "sparse.accumulate_gradients",
        {"group": "test_group", "dim": 4, "key_count": 2},
    )
    assert observed[1] == ("native_accumulate", 2)
