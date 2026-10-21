from __future__ import annotations

import cupy
import jax
import numpy as np
import pytest

from flare.kernels.backends.cpu_numba import CpuNumbaSparseEmbeddingOperator
from flare.kernels.backends.gpu_cupy import CupySparseEmbeddingOperator
from flare.kernels.backends.jax_backend import JaxSparseEmbeddingOperator


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


def test_cpu_scatter_promotes_bfloat16_transport_to_native_fp32() -> None:
    import ml_dtypes

    backend = CpuNumbaSparseEmbeddingOperator(
        value_dtype="float16", gradient_dtype="float32"
    )
    inverse = np.asarray([0, 1], dtype=np.int32)
    gradient = np.asarray([[1.5], [2.25]], dtype=ml_dtypes.bfloat16)
    result = backend.scatter(2, 1, inverse, 2, gradient)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.asarray([[1.5], [2.25]], dtype=np.float32))


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
