import numpy as np
import pytest
import sparse_embedding as se

from flare.runtime.sparse_storage import sparse_storage_options


def make_store(options, dim=4):
    return se.EmbeddingStore(
        se.EmbeddingConfig(dims=[dim], thread_budget=2, **options),
        initializer_config=se.Zeros(),
        optimizer_config=se.Adam(lr=0.1),
        score_policy="lfu",
    )


def test_explicit_page_capacity_allows_growth_past_native_limit():
    limited = make_store(
        {
            "storage": se.MemoryOnly(),
            "storage_tuning": se.StorageTuningOptions(
                entries_per_page=4096,
                max_pages=2,
                calibrate_page_geometry=False,
            ),
        }
    )
    with pytest.raises(
        (RuntimeError, ValueError), match="cohort allocation page capacity exceeded"
    ):
        for start in range(0, 20000, 1000):
            limited.lookup_or_insert(np.arange(start, start + 1000, dtype=np.int64), 4)
    expanded = make_store(
        sparse_storage_options({"sparse_storage_max_pages": 64}, group="test")
    )
    for start in range(0, 20000, 1000):
        expanded.lookup_or_insert(np.arange(start, start + 1000, dtype=np.int64), 4)
    assert expanded.total_size() == 20000


def test_spill_preserves_keys_values_and_optimizer_state(tmp_path):
    options = sparse_storage_options(
        {
            "sparse_storage_memory_limit_bytes": 8 * 1024**2,
            "sparse_storage_max_spill_bytes": 256 * 1024**2,
            "sparse_storage_spill_dir": str(tmp_path),
            "sparse_storage_max_pages": 128,
        },
        group="test",
    )
    store = make_store(options, dim=256)
    selected = np.arange(10, dtype=np.int64)
    store.lookup_or_insert(selected, 256)
    store.accumulate_gradients(selected, np.ones((10, 256), dtype=np.float32))
    store.optimizer_step(1)
    for start in range(0, 40000, 1000):
        store.lookup_or_insert(np.arange(start, start + 1000, dtype=np.int64), 256)
    stats = store.storage_stats()
    assert stats["spill_count"] > 0 and stats["spill_bytes"] > 0
    assert store.total_size() == 40000
    observed = {}
    for page in store.export_page_images():
        keys = np.asarray(page["keys"], dtype=np.uint64).view(np.int64)
        values = np.asarray(page["values"], dtype=np.float32).reshape(-1, 256)
        steps = np.asarray(page["steps"])
        optimizer = np.asarray(page["optimizer_states"], dtype=np.float32).reshape(
            len(keys), -1
        )
        for index in np.flatnonzero(keys < 10):
            observed[int(keys[index])] = (
                values[index].copy(),
                int(steps[index]),
                optimizer[index].copy(),
            )
    assert set(observed) == set(range(10))
    for value, step, optimizer in observed.values():
        np.testing.assert_allclose(value, -0.1, rtol=1e-5, atol=1e-6)
        assert step == 1 and np.isfinite(optimizer).all() and np.any(optimizer != 0)

    from sparse_embedding.checkpoint import (
        LocalLanceStorage,
        PartitionSpec,
        RestorePlan,
        SparseCheckpoint,
    )

    storage = LocalLanceStorage(tmp_path / "checkpoint")
    try:
        checkpoint = SparseCheckpoint(storage)
        saved = checkpoint.save_training_checkpoint(store, "spilled-training")
        restored = make_store(
            sparse_storage_options(
                {
                    "sparse_storage_memory_limit_bytes": 8 * 1024**2,
                    "sparse_storage_max_spill_bytes": 256 * 1024**2,
                    "sparse_storage_spill_dir": str(tmp_path / "restored-spill"),
                    "sparse_storage_max_pages": 128,
                },
                group="restored",
            ),
            dim=256,
        )
        checkpoint.load_training_checkpoint(
            restored,
            saved.handle,
            RestorePlan.from_partition(
                spec=PartitionSpec.modulo(),
                partition_count=1,
                partition_id=0,
                key_dtype="int64",
            ),
        )
        assert restored.total_size() == 40000
        restored.accumulate_gradients(selected, np.ones((10, 256), dtype=np.float32))
        restored.optimizer_step(1)
        values, _ = restored.lookup(selected, 256)
        np.testing.assert_allclose(np.from_dlpack(values), -0.2, rtol=1e-5, atol=1e-6)
    finally:
        storage.close()
