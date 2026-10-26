from unittest.mock import Mock

import numpy as np
import pytest
import sparse_embedding as se

from tools.validation import audit_sparse_checkpoint


@pytest.fixture
def store():
    return se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[2], thread_budget=1, value_dtype="float32", key_dtype="int64"
        ),
        initializer_config=se.Zeros(),
        optimizer_config=se.Adam(lr=0.1),
    )


def test_native_sparse_audit_reads_optimizer_steps_and_finite_state(store):
    keys = np.asarray([3, -7], dtype=np.int64)
    store.lookup_or_insert(keys, 2)
    store.accumulate_gradients(keys, np.ones((2, 2), dtype=np.float32))
    store.optimizer_step(1)

    result = audit_sparse_checkpoint.summarize_store(store, expected_optimizer_step=1)
    assert result["entry_count"] == 2
    assert result["dimension_slots"] == 1
    assert result["optimizer_elements_per_key"] == {2: 4}
    assert result["key_step_min"] == result["key_step_max"] == 1
    assert result["fingerprints"]["values"]["nan_count"] == 0
    assert result["fingerprints"]["optimizer"]["inf_count"] == 0


def test_native_sparse_audit_rejects_non_finite_model_state(store):
    store.assign(
        np.asarray([3], dtype=np.int64),
        np.asarray([[np.nan, 1.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        audit_sparse_checkpoint.summarize_store(store, expected_optimizer_step=1)


def test_native_sparse_audit_rejects_extra_optimizer_updates(store):
    keys = np.asarray([3], dtype=np.int64)
    store.lookup_or_insert(keys, 2)
    for _ in range(2):
        store.accumulate_gradients(keys, np.ones((1, 2), dtype=np.float32))
        store.optimizer_step(1)
    with pytest.raises(ValueError, match="per-key update count"):
        audit_sparse_checkpoint.summarize_store(store, expected_optimizer_step=1)


def test_native_sparse_audit_rejects_page_lengths_that_cancel_after_concatenation(
    store,
):
    store.lookup_or_insert(np.asarray([3, 7], dtype=np.int64), 2)
    image = store.export_page_images()[0]
    pages = [
        {
            "embedding_dim": 2,
            "keys": image["keys"][:1],
            "values": image["values"][:2],
            "optimizer_states": image["optimizer_states"][:4],
            "steps": [0, 0],
        },
        {
            "embedding_dim": 2,
            "keys": image["keys"][1:],
            "values": image["values"][2:],
            "optimizer_states": image["optimizer_states"][4:],
            "steps": [],
        },
    ]
    exported_store = Mock(wraps=store)
    exported_store.export_page_images.return_value = pages
    with pytest.raises(ValueError, match="per-page"):
        audit_sparse_checkpoint.summarize_store(
            exported_store, expected_optimizer_step=1
        )
