from __future__ import annotations

import cupy
import jax
import numpy as np
import pytest

from flare.kernels.backends.cpu_numba import CpuNumbaSparseEmbeddingOperator
from flare.kernels.backends.gpu_cupy import CupySparseEmbeddingOperator
from flare.kernels.backends.jax_backend import JaxSparseEmbeddingOperator
from flare.runtime.sparse_precision import prepare_sparse_native_gradient


class _Stream:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.events.append("exit")


class _OperationStream:
    def __init__(self) -> None:
        self.active = False
        self.events: list[str] = []

    def __enter__(self):
        assert not self.active
        self.active = True
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self.active
        self.events.append("exit")
        self.active = False


class _OperationArray:
    def __init__(
        self, shape: tuple[int, ...], size: int, events: list[str], dtype=np.float32
    ) -> None:
        self.shape = shape
        self.size = size
        self.dtype = dtype
        self.events = events
        self.data = type("Data", (), {"ptr": 0})()

    def reshape(self, *_shape):
        self.events.append("reshape")
        return self

    def astype(self, *_args, **_kwargs):
        self.events.append("cast")
        return self


class _OperationCupy:
    float32 = np.float32
    float16 = np.float16
    int32 = np.int32

    def __init__(self, stream: _OperationStream, events: list[str]) -> None:
        self.stream = stream
        self.events = events

    def empty(self, shape, *, dtype):
        assert self.stream.active
        assert dtype is self.float32
        self.events.append("empty")
        return _OperationArray(tuple(shape), int(np.prod(shape)), self.events)


def test_jax_backend_converts_cupy_before_pooled_gather() -> None:
    backend = JaxSparseEmbeddingOperator(
        value_dtype="float32", gradient_dtype="float32"
    )
    embedding = cupy.arange(8, dtype=cupy.float32).reshape(4, 2)
    inverse = np.asarray([0, 4, 2, 1], dtype=np.int32)
    result = backend.gather_pooled(embedding, inverse, 4, 2, (1, 1), 4)
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result)).reshape(4, 2),
        np.asarray([[0.0, 1.0], [0.0, 0.0], [4.0, 5.0], [2.0, 3.0]], dtype=np.float32),
    )


def test_jax_backend_rejects_non_dlpack_embedding() -> None:
    backend = JaxSparseEmbeddingOperator(
        value_dtype="float32", gradient_dtype="float32"
    )
    with pytest.raises(TypeError, match="DLPack|array"):
        backend.gather_pooled(
            object(), np.asarray([0, 1, 2, 3], dtype=np.int32), 4, 2, (1, 1), 4
        )


def test_cupy_imports_cuda_jax_gradient_without_host_conversion(monkeypatch):
    backend = CupySparseEmbeddingOperator("float16", "float32")
    source = jax.device_put(jax.numpy.arange(64, dtype=jax.numpy.bfloat16))
    if source.__dlpack_device__()[0] != 2:
        pytest.skip("requires CUDA JAX backend")

    class CudaJaxValue:
        def __jax_array__(self):
            return source

        def __dlpack_device__(self):
            return source.__dlpack_device__()

        def __dlpack__(self, **kwargs):
            return source.__dlpack__(**kwargs)

        def __array__(self, *args, **kwargs):
            raise AssertionError("CUDA gradient must not be copied through NumPy")

    actual = backend.array(CudaJaxValue())
    backend.stream.synchronize()
    assert actual.data.ptr == source.unsafe_buffer_pointer()
    assert actual.dtype == cupy.dtype("bfloat16")
    np.testing.assert_array_equal(
        cupy.asnumpy(actual).astype(np.float32), np.arange(64)
    )


def test_sparse_backend_requires_explicit_gradient_dtype() -> None:
    with pytest.raises(TypeError):
        JaxSparseEmbeddingOperator(value_dtype="float16")


@pytest.mark.parametrize(
    "operator",
    [JaxSparseEmbeddingOperator, CpuNumbaSparseEmbeddingOperator],
)
def test_sparse_backend_rejects_fp16_gradient_dtype(operator) -> None:
    with pytest.raises(ValueError, match="float32"):
        operator(value_dtype="float16", gradient_dtype="float16")


def test_jax_fp16_pooled_gather_preserves_dtype_and_ieee_overflow() -> None:
    backend = JaxSparseEmbeddingOperator(
        value_dtype="float32", gradient_dtype="float32"
    )
    embedding = jax.numpy.asarray([[1.5], [2.0]], dtype=jax.numpy.float16)
    inverse = jax.numpy.asarray([0, 1], dtype=jax.numpy.int32)
    result = backend.gather_pooled(embedding, inverse, 1, 1, (1, 2), 2)
    assert result.dtype == jax.numpy.float16
    np.testing.assert_array_equal(
        np.asarray(result), np.asarray([[[3.5]]], dtype=np.float16)
    )

    overflow = jax.numpy.asarray([[65504.0], [65504.0]], dtype=jax.numpy.float16)
    overflow_result = backend.gather_pooled(overflow, inverse, 1, 1, (1, 2), 2)
    assert overflow_result.dtype == jax.numpy.float16
    assert np.isinf(np.asarray(overflow_result)).all()


def test_cpu_fp16_pooled_gather_preserves_dtype_and_accumulates_safely() -> None:
    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float32", gradient_dtype="float32"
    )
    embedding = np.asarray([[1.5, 2.0], [3.0, 4.0]], dtype=np.float16)
    inverse = np.asarray([0, 1, 2, 2], dtype=np.int32)
    result = backend.gather_pooled(embedding, inverse, 2, 2, (1, 2), 2)
    assert result.dtype == np.float16
    np.testing.assert_array_equal(
        result, np.asarray([[[4.5, 6.0]], [[0.0, 0.0]]], dtype=np.float16)
    )


def test_cpu_fp16_pooling_uses_wide_accumulator_before_output_cast() -> None:
    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float32", gradient_dtype="float32"
    )
    embedding = np.asarray([[40000.0], [-40000.0], [1.0]], dtype=np.float16)
    inverse = np.asarray([0, 1, 2], dtype=np.int32)
    result = backend.gather_pooled(embedding, inverse, 1, 1, (1, 3), 3)
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, np.asarray([[[1.0]]], dtype=np.float16))


def test_cpu_scatter_exposes_declared_gradient_dtype() -> None:
    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float16", gradient_dtype="float32"
    )
    inverse = np.asarray([0, 1], dtype=np.int32)
    gradient = np.asarray([[1.5], [2.25]], dtype=np.float16)
    result = backend.scatter(2, 1, inverse, 2, gradient)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.asarray([[1.5], [2.25]], dtype=np.float32))


def test_cpu_bfloat16_transport_survives_native_scatter_boundary() -> None:
    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float16", gradient_dtype="float32"
    )
    inverse = np.asarray([0, 1], dtype=np.int32)
    gradient = jax.numpy.asarray([[1.5], [2.25]], dtype=jax.numpy.bfloat16)
    transported = backend.local_array(gradient)
    assert transported.dtype == np.dtype("bfloat16")
    native = prepare_sparse_native_gradient(
        transported, transport_dtype="bfloat16", native_dtype="float32"
    )
    result = backend.scatter(2, 1, inverse, 2, native)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.asarray([[1.5], [2.25]], dtype=np.float32))


def test_cpu_array_preserves_bfloat16_host_dtype() -> None:
    import ml_dtypes

    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float16", gradient_dtype="float32"
    )
    source = np.asarray([[1.5, 2.25]], dtype=ml_dtypes.bfloat16)
    result = backend.array(source)
    assert result.dtype == np.dtype("bfloat16")
    assert result is source


def test_cpu_fp16_pooled_gather_preserves_ieee_overflow() -> None:
    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float32", gradient_dtype="float32"
    )
    embedding = np.asarray([[65504.0], [65504.0]], dtype=np.float16)
    inverse = np.asarray([0, 1], dtype=np.int32)
    result = backend.gather_pooled(embedding, inverse, 1, 1, (1, 2), 2)
    assert result.dtype == np.float16
    assert np.isinf(result).all()


def test_cupy_empty_fp16_pooled_gather_skips_zero_block_launch() -> None:
    stream = _OperationStream()
    events: list[str] = []

    class _EmptyCupy:
        float16 = np.float16
        float32 = np.float32
        int32 = np.int32

        def empty(self, shape, *, dtype):
            events.append(f"empty:{dtype}")
            return _EmptyArray(tuple(shape), int(np.prod(shape)), events, dtype)

    class _EmptyArray(_OperationArray):
        def reshape(self, *shape):
            self.events.append("reshape")
            self.shape = tuple(
                0 if dimension == -1 else dimension for dimension in shape
            )
            return self

    class _NoLaunch:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("empty pooled gather must not launch a kernel")

    backend = object.__new__(CupySparseEmbeddingOperator)
    backend.stream = stream
    backend.cp = _EmptyCupy()
    backend._block_size = 256
    backend.gather_fp16_kernel = _NoLaunch()
    backend.gather_pooled_fp16_kernel = _NoLaunch()
    backend._array_on_stream = lambda value: value
    backend._inverse_on_stream = lambda value: value

    embedding = _EmptyArray((0, 2), 0, events, np.float16)
    inverse = _EmptyArray((0,), 0, events, np.int32)
    result = backend.gather_pooled(embedding, inverse, 0, 2, (1, 2), 0)

    assert result.shape == (0, 1, 2)
    assert result.dtype == np.float16
    assert stream.events == ["enter", "exit"]

    gather_result = backend.gather(embedding, inverse, 0, 2, 0)
    assert gather_result.shape == (0, 0, 2)
    assert gather_result.dtype == np.float16


def test_cupy_sparse_tensor_backend_enters_its_nonblocking_stream_for_conversion(
    monkeypatch,
) -> None:
    backend = object.__new__(CupySparseEmbeddingOperator)
    backend.stream = _Stream()
    backend._backend = type(
        "Backend",
        (),
        {
            "as_tensor": lambda _self, value, *, device: (
                "device-array" if device == "cuda" else None
            )
        },
    )()
    observed: dict[str, object] = {}

    original = backend._backend.as_tensor

    def convert(value, *, device):
        observed["value"] = value
        observed["stream_active"] = backend.stream.events == ["enter"]
        return original(value, device=device)

    monkeypatch.setattr(backend._backend, "as_tensor", convert)

    assert backend.array("host-array") == "device-array"
    assert observed == {"value": "host-array", "stream_active": True}
    assert backend.stream.events == ["enter", "exit"]


def test_cupy_gather_pooled_orders_conversion_and_launch_on_one_stream() -> None:
    stream = _OperationStream()
    events: list[str] = []
    backend = object.__new__(CupySparseEmbeddingOperator)
    backend.stream = stream
    backend.cp = _OperationCupy(stream, events)
    backend._block_size = 256
    embedding = _OperationArray((3, 3), 9, events)
    inverse = _OperationArray((2, 1, 2), 4, events)

    def convert(value):
        assert stream.active
        events.append(f"convert:{value}")
        return embedding if value == "embedding" else inverse

    def launch(*_args, **kwargs) -> None:
        assert kwargs["stream"] is stream
        assert stream.active
        events.append("launch")

    backend.gather_pooled_kernel = launch
    backend._backend = type(
        "Backend",
        (),
        {"as_tensor": lambda _self, value, *, device: convert(value)},
    )()
    backend.gather_pooled("embedding", "inverse", 2, 3, (1, 2), 3)

    assert stream.events == ["enter", "exit"]
    assert events == [
        "convert:embedding",
        "reshape",
        "convert:inverse",
        "cast",
        "reshape",
        "empty",
        "launch",
    ]


@pytest.mark.parametrize(
    "feature_count,max_elements,dim",
    [(12, 30, 128), (44, 30, 40), (2, 1, 418)],
)
def test_cupy_register_scatter_matches_duplicate_key_sum(
    feature_count: int, max_elements: int, dim: int
) -> None:
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")
    batch_size, rows = 48, 256
    rng = np.random.default_rng(93)
    inverse = rng.integers(
        0, rows // 4, (batch_size, feature_count, max_elements), dtype=np.int32
    )
    inverse[rng.random(inverse.shape) < 0.8] = rows
    gradient = rng.standard_normal((batch_size, feature_count, dim)).astype(np.float32)
    flat_inverse = inverse.reshape(-1)
    flat_gradient = np.repeat(gradient.reshape(-1, dim), max_elements, axis=0)
    valid = flat_inverse != rows
    expected = np.zeros((rows, dim), dtype=np.float32)
    np.add.at(expected, flat_inverse[valid], flat_gradient[valid])

    operator = CupySparseEmbeddingOperator("float16", "float32")
    inverse_device = cupy.asarray(inverse)
    gradient_device = cupy.asarray(gradient)
    cupy.cuda.Stream.null.synchronize()
    actual = operator.scatter_pooled(
        rows,
        dim,
        inverse_device,
        rows,
        gradient_device,
        (feature_count, max_elements),
    )
    operator.stream.synchronize()
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, atol=2e-4, rtol=2e-4)


def test_cupy_pooled_scatter_rejects_partial_inverse_slots() -> None:
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")
    operator = CupySparseEmbeddingOperator("float16", "float32")
    with pytest.raises(ValueError, match="inverse slots must match"):
        operator.scatter_pooled(
            2,
            2,
            cupy.asarray([0, 1, 2], dtype=cupy.int32),
            2,
            cupy.ones((1, 1, 2), dtype=cupy.float32),
            (1, 2),
        )
