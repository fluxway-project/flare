from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flare.runtime.fluxway.transport import make_transport_tensor
from recipes.recsys.ranking.workers.train_worker import RecsysBackboneTrainSPMD


@pytest.mark.parametrize("group_count", [1, 3])
def test_cupy_scatter_finishes_before_bf16_gradient_is_routed(
    group_count: int,
) -> None:
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")

    from flare.kernels.backends.gpu_cupy import CupySparseEmbeddingOperator

    batch_size, feature_count, max_elements, width, rows = 2560, 4, 4, 32, 512
    rng = np.random.default_rng(13)
    inverse = rng.integers(
        0, rows + 1, (batch_size, feature_count, max_elements), dtype=np.int32
    )
    counts = np.bincount(inverse[inverse != rows], minlength=rows)
    groups = tuple(f"group_{index}" for index in range(group_count))

    worker = object.__new__(RecsysBackboneTrainSPMD)
    worker.ctx = SimpleNamespace(step=0)
    worker.sparse_groups = groups
    worker.sparse_group_widths = {
        group: (feature_count, max_elements) for group in groups
    }
    worker.sparse_op = CupySparseEmbeddingOperator("float16", "float32")
    worker.sparse_gradient_dtype = "float32"
    worker.sparse_gradient_transport_dtype = "bfloat16"
    worker.execution_tracer = None
    worker.validate_tensor_values = False
    inputs = {}
    for index, group in enumerate(groups):
        inputs[f"grad_jax_{group}"] = make_transport_tensor(
            cupy.full(
                (batch_size, feature_count, width),
                1 / (256 * (2**index)),
                dtype=cupy.dtype("bfloat16"),
            )
        )
        inputs[f"padded_inverse_{group}"] = make_transport_tensor(cupy.asarray(inverse))
        inputs[f"emb_ref_{group}"] = make_transport_tensor(
            cupy.zeros((rows, width), dtype=cupy.float16)
        )
    for _ in range(10):
        result = worker.scatter_sparse_grads(**inputs)
        for index, group in enumerate(groups):
            gradient = result[f"grad_{group}"].flat_values
            expected = np.broadcast_to(
                (counts / (256 * (2**index)))[:, None], (rows, width)
            )
            assert gradient.dtype == cupy.dtype("bfloat16")
            np.testing.assert_array_equal(
                cupy.asnumpy(gradient).astype(np.float32), expected
            )
