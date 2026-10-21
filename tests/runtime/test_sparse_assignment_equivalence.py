from pathlib import Path

import cupy as cp
import jax
import numpy as np
import pytest
import sparse_embedding as se
from fluxway.common.tensor_backend import get_backend, use_backend
from fluxway.common.types.sharding import IndexedModuloPartition
from sparse_embedding.checkpoint import (
    LocalLanceStorage,
    PartitionSpec,
    RestorePlan,
    SparseCheckpoint,
)


@pytest.fixture(params=("int32", "uint32", "int64", "uint64"))
def checkpoint_case(request, tmp_path: Path):
    key_dtype = request.param
    bounds = np.iinfo(key_dtype)
    values = {int(bounds.min), int(bounds.max), 0, 1, 2, 5}
    if bounds.min < 0:
        values.update((-5, -1))
    else:
        values.add(1 << (bounds.bits - 1))
    keys = np.asarray(sorted(values), dtype=key_dtype)
    config = se.EmbeddingConfig(
        dims=[2],
        thread_budget=1,
        value_dtype="float32",
        key_dtype=key_dtype,
        enable_optimizer=True,
    )
    source = se.EmbeddingStore(config, optimizer_config=se.Adam(lr=0.01))
    embeddings = np.arange(keys.size * 2, dtype=np.float32).reshape(-1, 2)
    source.assign(keys, embeddings)
    storage = LocalLanceStorage(tmp_path / key_dtype)
    try:
        checkpoint = SparseCheckpoint(storage)
        saved = checkpoint.save_training_checkpoint(source, "assignment-vectors")
        yield key_dtype, keys, embeddings, config, checkpoint, saved
    finally:
        storage.close()


@pytest.mark.parametrize("partition_count", (1, 2, 3, 7))
def test_native_restore_matches_runtime_backends(checkpoint_case, partition_count):
    key_dtype, keys, embeddings, config, checkpoint, saved = checkpoint_case
    expected = np.asarray([int(key) % partition_count for key in keys], dtype=np.uint32)
    native = np.from_dlpack(se.ModuloPartitioner(partition_count).partition(keys))
    np.testing.assert_array_equal(native, expected)

    for backend_name in ("numba_cpu", "jax", "cupy"):
        with jax.enable_x64(True), use_backend(backend_name):
            backend = get_backend()
            runtime_keys = backend.tensor(keys, dtype=keys.dtype)
            array_type = {"numba_cpu": np.ndarray, "jax": jax.Array, "cupy": cp.ndarray}
            assert isinstance(runtime_keys, array_type[backend_name])
            assert np.dtype(runtime_keys.dtype) == keys.dtype
            runtime = IndexedModuloPartition()(
                runtime_keys, dim_size=keys.size, num_workers=partition_count
            )
            np.testing.assert_array_equal(backend.to_numpy(runtime), expected)

    for partition_id in range(partition_count):
        target = se.EmbeddingStore(config, optimizer_config=se.Adam(lr=0.01))
        options = RestorePlan.from_partition(
            spec=PartitionSpec.modulo(),
            partition_count=partition_count,
            partition_id=partition_id,
            key_dtype=key_dtype,
        )
        checkpoint.load_training_checkpoint(target, saved.handle, options)
        selected = expected == partition_id
        np.testing.assert_array_equal(np.from_dlpack(target.contains(keys)), selected)
        assert target.total_size() == np.count_nonzero(selected)
        if np.any(selected):
            restored, _ = target.lookup(keys[selected], dim=2)
            np.testing.assert_array_equal(
                np.from_dlpack(restored).reshape(-1, 2), embeddings[selected]
            )
