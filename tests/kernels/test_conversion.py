from __future__ import annotations

import os
import sys
from typing import Any

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import cupy as cp
import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import pytest

from flare.kernels.conversion import (
    to_cupy_array,
    to_host_array,
    to_jax_array,
    to_jax_tree,
)


class _Stream:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.events.append("exit")


class _FakeCupyArray:
    pass


class _FakeCupy:
    ndarray = _FakeCupyArray

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object | None]] = []

    def asarray(self, value, *, blocking: bool):
        self.calls.append(("asarray", value, blocking))
        return _FakeCupyArray()

    def from_dlpack(self, value):
        self.calls.append(("from_dlpack", value, None))
        return _FakeCupyArray()


class _HostTransferStream:
    def __init__(self, non_blocking: bool) -> None:
        self.non_blocking = non_blocking


class _HostTransferEvent:
    def __init__(self) -> None:
        self.recorded_stream: _HostTransferStream | None = None
        self.synchronized = False

    def record(self, stream: _HostTransferStream) -> None:
        self.recorded_stream = stream

    def synchronize(self) -> None:
        self.synchronized = True


class _HostTransferArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.dtype = values.dtype
        self.shape = values.shape
        self.size = values.size
        self.nbytes = values.nbytes
        self.get_calls: list[tuple[_HostTransferStream, bool]] = []

    def get(
        self,
        *,
        stream: _HostTransferStream,
        out: np.ndarray,
        blocking: bool,
    ) -> np.ndarray:
        self.get_calls.append((stream, blocking))
        out[...] = self.values
        return out


class _HostTransferCuda:
    def __init__(self) -> None:
        self.streams: list[_HostTransferStream] = []
        self.events: list[_HostTransferEvent] = []
        self.pinned_allocations: list[bytearray] = []

    def Stream(self, *, non_blocking: bool) -> _HostTransferStream:
        stream = _HostTransferStream(non_blocking)
        self.streams.append(stream)
        return stream

    def Event(self) -> _HostTransferEvent:
        event = _HostTransferEvent()
        self.events.append(event)
        return event

    def alloc_pinned_memory(self, size: int) -> bytearray:
        allocation = bytearray(size)
        self.pinned_allocations.append(allocation)
        return allocation


class _HostTransferCupy:
    ndarray = _HostTransferArray

    def __init__(self) -> None:
        self.cuda = _HostTransferCuda()


class _CpuDLPack:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __dlpack__(self, *args, **kwargs):
        return self.values.__dlpack__(*args, **kwargs)

    @staticmethod
    def __dlpack_device__() -> tuple[int, int]:
        return (1, 0)


class _CudaDLPack:
    @staticmethod
    def __dlpack__():
        return object()

    @staticmethod
    def __dlpack_device__() -> tuple[int, int]:
        return (2, 0)


class _UnsupportedDLPack:
    @staticmethod
    def __dlpack__():
        return object()

    @staticmethod
    def __dlpack_device__() -> tuple[int, int]:
        return (7, 0)


class _CudaArrayInterfaceObject:
    def __init__(self, cupy_arr: cp.ndarray) -> None:
        self._arr = cupy_arr
        self.__cuda_array_interface__ = cupy_arr.__cuda_array_interface__

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    def __getitem__(self, item):
        return _CudaArrayInterfaceObject(self._arr[item])


def _to_cuda_array(data: Any) -> _CudaArrayInterfaceObject:
    return _CudaArrayInterfaceObject(cp.asarray(data))


def _get_cuda_ptr(arr: Any) -> int | None:
    if hasattr(arr, "unsafe_buffer_pointer"):
        return arr.unsafe_buffer_pointer()
    if hasattr(arr, "data") and hasattr(arr.data, "ptr"):
        return arr.data.ptr
    if hasattr(arr, "device_ctypes_pointer"):
        return arr.device_ctypes_pointer.value
    if hasattr(arr, "__cuda_array_interface__"):
        return arr.__cuda_array_interface__["data"][0]
    return None


@pytest.fixture
def fake_cupy(monkeypatch) -> _FakeCupy:
    fake = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return fake


def test_cpu_numpy_conversion_queues_nonblocking_copy_on_supplied_stream(
    fake_cupy,
) -> None:
    stream = _Stream()
    source = np.asarray([1, 2, 3], dtype=np.int32)

    result = to_cupy_array(source, stream=stream)

    assert isinstance(result, _FakeCupyArray)
    assert stream.events == ["enter", "exit"]
    assert len(fake_cupy.calls) == 1
    method, host_values, blocking = fake_cupy.calls[0]
    assert method == "asarray"
    assert blocking is False
    np.testing.assert_array_equal(host_values, source)


def test_cpu_dlpack_conversion_queues_nonblocking_copy_on_supplied_stream(
    fake_cupy,
) -> None:
    stream = _Stream()
    source = _CpuDLPack(np.asarray([1, 2, 3], dtype=np.int32))

    result = to_cupy_array(source, stream=stream)

    assert isinstance(result, _FakeCupyArray)
    assert stream.events == ["enter", "exit"]
    assert len(fake_cupy.calls) == 1
    method, host_values, blocking = fake_cupy.calls[0]
    assert method == "asarray"
    assert blocking is False
    np.testing.assert_array_equal(host_values, np.asarray([1, 2, 3], dtype=np.int32))


def test_cuda_dlpack_conversion_uses_zero_copy_import_on_supplied_stream(
    fake_cupy,
) -> None:
    stream = _Stream()
    source = _CudaDLPack()

    result = to_cupy_array(source, stream=stream)

    assert isinstance(result, _FakeCupyArray)
    assert stream.events == ["enter", "exit"]
    assert fake_cupy.calls == [("from_dlpack", source, None)]


def test_unsupported_dlpack_device_raises_without_conversion(fake_cupy) -> None:
    with pytest.raises(TypeError, match="unsupported DLPack device type"):
        to_cupy_array(_UnsupportedDLPack())
    assert fake_cupy.calls == []


def test_cupy_host_transfer_uses_pinned_nonblocking_copy_and_event(monkeypatch) -> None:
    fake = _HostTransferCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    source = _HostTransferArray(np.asarray([[1.0, 2.0]], dtype=np.float32))

    result = to_host_array(source, dtype=np.float32)

    np.testing.assert_array_equal(result, source.values)
    assert len(fake.cuda.streams) == 1
    assert fake.cuda.streams[0].non_blocking is True
    assert source.get_calls == [(fake.cuda.streams[0], False)]
    assert len(fake.cuda.pinned_allocations) == 1
    assert len(fake.cuda.events) == 1
    assert fake.cuda.events[0].recorded_stream is fake.cuda.streams[0]
    assert fake.cuda.events[0].synchronized is True


def test_cupy_to_jax_zero_copy() -> None:
    cp_arr = cp.arange(16, dtype=cp.float32)
    jax_arr = to_jax_array(cp_arr)
    assert isinstance(jax_arr, jax.Array)
    assert _get_cuda_ptr(cp_arr) == _get_cuda_ptr(jax_arr)
    np.testing.assert_allclose(np.asarray(jax.device_get(jax_arr)), np.arange(16))


def test_cuda_interface_to_jax_zero_copy() -> None:
    cuda_arr = _to_cuda_array(np.arange(16, dtype=np.float32))
    jax_arr = to_jax_array(cuda_arr)
    assert isinstance(jax_arr, jax.Array)
    assert _get_cuda_ptr(cuda_arr) == _get_cuda_ptr(jax_arr)
    np.testing.assert_allclose(np.asarray(jax.device_get(jax_arr)), np.arange(16))


def test_jax_identity() -> None:
    orig = jnp.arange(16, dtype=jnp.float32)
    res = to_jax_array(orig)
    assert res is orig


def test_readonly_transport_tensor_is_imported_through_coherence_seam() -> None:
    from fluxway.common.types.tensor import TransportTensor

    source = np.arange(32, dtype=np.float32)
    tensor = TransportTensor(source).zero_copy_readonly_alias()

    result = to_jax_array(tensor.flat_values)

    np.testing.assert_array_equal(np.asarray(jax.device_get(result)), source)
    assert not tensor.flat_values.flags.writeable


def test_readonly_numpy_input_is_copied_only_by_standard_opportunistic_policy() -> None:
    source = np.arange(32, dtype=np.float32)
    source.flags.writeable = False

    result = to_jax_array(source)

    np.testing.assert_array_equal(np.asarray(jax.device_get(result)), source)


def test_payload_container_conversion() -> None:
    cp_arr = cp.arange(8, dtype=cp.float32)
    cuda_arr = _to_cuda_array(np.arange(8, dtype=np.float32))

    jax_cp = to_jax_array(cp_arr)
    jax_cuda = to_jax_array(cuda_arr)

    assert _get_cuda_ptr(cp_arr) == _get_cuda_ptr(jax_cp)
    assert _get_cuda_ptr(cuda_arr) == _get_cuda_ptr(jax_cuda)


def test_nested_container_seamless_conversion() -> None:
    cp_arr = cp.arange(4, dtype=cp.float32)
    cuda_arr = _to_cuda_array(np.arange(4, dtype=np.float32))

    batch = {
        "geek": cp_arr,
        "job": cuda_arr,
        "items": [cp_arr, cuda_arr],
        "tuple_items": (cp_arr, cuda_arr),
    }

    converted = to_jax_tree(batch)

    assert isinstance(converted["geek"], jax.Array)
    assert isinstance(converted["job"], jax.Array)
    assert isinstance(converted["items"][0], jax.Array)
    assert isinstance(converted["items"][1], jax.Array)
    assert isinstance(converted["tuple_items"][0], jax.Array)
    assert isinstance(converted["tuple_items"][1], jax.Array)

    assert _get_cuda_ptr(cp_arr) == _get_cuda_ptr(converted["geek"])
    assert _get_cuda_ptr(cuda_arr) == _get_cuda_ptr(converted["job"])
    assert _get_cuda_ptr(cp_arr) == _get_cuda_ptr(converted["items"][0])
    assert _get_cuda_ptr(cuda_arr) == _get_cuda_ptr(converted["items"][1])
    assert _get_cuda_ptr(cp_arr) == _get_cuda_ptr(converted["tuple_items"][0])
    assert _get_cuda_ptr(cuda_arr) == _get_cuda_ptr(converted["tuple_items"][1])


def test_non_contiguous_gpu_arrays_resolved_on_device() -> None:
    cp_arr = cp.arange(10, dtype=cp.float32)[::2]
    cuda_arr = _to_cuda_array(np.arange(10, dtype=np.float32))[::2]

    jax_cp = to_jax_array(cp_arr)
    jax_cuda = to_jax_array(cuda_arr)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(jax_cp)), np.array([0, 2, 4, 6, 8])
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(jax_cuda)), np.array([0, 2, 4, 6, 8])
    )


def test_unsupported_type_raises_typeerror_without_fallback() -> None:
    with pytest.raises(TypeError):
        to_jax_array("unsupported_string")

    with pytest.raises(TypeError):
        to_jax_array(object())


def test_open_closed_extensibility_via_fluxway_subclass() -> None:
    from fluxway.common.tensor_backend import JaxBackend

    from flare.kernels import conversion as conv_mod

    class _CustomGPUBuffer:
        def __init__(self, data: Any) -> None:
            self.internal_data = data

    class _ExtendedBackend(JaxBackend):
        def asarray(self, x, *, copy=None, device=None):
            if isinstance(x, _CustomGPUBuffer):
                return super().asarray(x.internal_data, copy=copy, device=device)
            return super().asarray(x, copy=copy, device=device)

    original_backend = conv_mod._jax_backend
    conv_mod._jax_backend = _ExtendedBackend()
    try:
        cuda_arr = _to_cuda_array(np.arange(6, dtype=np.float32))
        custom_buf = _CustomGPUBuffer(cuda_arr)
        jax_res = to_jax_array(custom_buf)

        assert isinstance(jax_res, jax.Array)
        assert _get_cuda_ptr(cuda_arr) == _get_cuda_ptr(jax_res)
    finally:
        conv_mod._jax_backend = original_backend


def test_to_host_array_supports_cuda_interface_payload() -> None:
    cuda_arr = _to_cuda_array(np.arange(5, dtype=np.float32))

    host_arr = to_host_array(cuda_arr)
    assert isinstance(host_arr, np.ndarray)
    np.testing.assert_allclose(host_arr, np.arange(5))
