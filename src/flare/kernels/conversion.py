from __future__ import annotations

import threading
from typing import Any

import numpy as np
from fluxway.common.tensor_backend import JaxBackend, as_native_zerocopy

_host_transfer_state = threading.local()

_DLPACK_DEVICE_CPU = 1
_DLPACK_DEVICE_CUDA = 2

_jax_backend = JaxBackend()


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "__array__"):
        return np.asarray(value)
    if hasattr(value, "addressable_shards"):
        shards = [np.asarray(shard.data) for shard in value.addressable_shards]
        if not shards:
            raise ValueError("Shard collection is empty.")
        return np.concatenate(shards, axis=0)
    return np.asarray(value)


def to_host_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=dtype)
    try:
        import cupy
    except ModuleNotFoundError:
        if hasattr(value, "__cuda_array_interface__"):
            raise TypeError(
                "CUDA-to-host conversion requires CuPy for a CUDA array source"
            ) from None
        return np.asarray(value, dtype=dtype)
    if hasattr(value, "__cuda_array_interface__") and not isinstance(
        value, cupy.ndarray
    ):
        value = cupy.asarray(value)
    if not isinstance(value, cupy.ndarray):
        return np.asarray(value, dtype=dtype)
    state = getattr(_host_transfer_state, "value", None)
    if state is None or state[0] is not cupy:
        state = (cupy, cupy.cuda.Stream(non_blocking=True))
        _host_transfer_state.value = state
    stream = state[1]
    allocation = cupy.cuda.alloc_pinned_memory(value.nbytes)
    result = np.ndarray(value.shape, dtype=value.dtype, buffer=allocation)
    value.get(stream=stream, out=result, blocking=False)
    event = cupy.cuda.Event()
    event.record(stream)
    event.synchronize()
    return np.asarray(result, dtype=dtype)


def to_dlpack(value: Any) -> Any:
    if hasattr(value, "__dlpack__"):
        return value.__dlpack__()
    raise TypeError(f"Value of type {type(value).__name__} does not support DLPack.")


def to_jax_array(value: Any) -> Any:
    """Import any GPU/CPU/container array into JAX, preferring zero-copy.

    Delegates to fluxway JaxBackend.asarray(copy=None) which handles:
    - JAX identity (no-op)
    - CuPy via DLPack
    - Numba DeviceNDArray via __cuda_array_interface__ -> CuPy -> DLPack
    - DLPack-compatible arrays without a host transfer when possible
    - dict / list / tuple recursive traversal
    """
    return _jax_backend.asarray(value, copy=None)


def to_jax_tree(value: Any) -> Any:
    """Recursively import all leaf arrays in a nested container into JAX."""
    return _jax_backend.asarray_tree(value, copy=None)


def to_jax_array_with_lease(
    value: Any, *, copy: bool | None = None
) -> tuple[Any, Any | None]:
    """Import one value and return its explicit Fluxway consumer lease."""
    return _jax_backend.asarray_with_lease(value, copy=copy)


def release_jax_leases(leases: Any) -> None:
    for lease in leases:
        lease.release()


def to_cupy_array(value: Any, *, stream: Any | None = None) -> Any:
    import cupy

    def on_stream(convert: Any) -> Any:
        if stream is None:
            return convert()
        with stream:
            return convert()

    if isinstance(value, cupy.ndarray):
        return value
    if hasattr(value, "__cuda_array_interface__") and not hasattr(value, "__dlpack__"):
        return on_stream(lambda: cupy.asarray(value))
    if hasattr(value, "__dlpack__"):
        if not hasattr(value, "__dlpack_device__"):
            raise TypeError(
                "CuPy DLPack conversion requires __dlpack_device__ to identify "
                f"the source device, got {type(value).__name__}."
            )
        device_type, _device_id = value.__dlpack_device__()
        if device_type == _DLPACK_DEVICE_CUDA:
            return on_stream(lambda: cupy.from_dlpack(value))
        if device_type == _DLPACK_DEVICE_CPU:
            return on_stream(
                lambda: cupy.asarray(np.from_dlpack(value), blocking=False)
            )
        raise TypeError(
            "CuPy conversion received unsupported DLPack device type "
            f"{device_type!r} from {type(value).__name__}."
        )
    return on_stream(lambda: cupy.asarray(value, blocking=False))


def extract_local_array(value: Any) -> Any:
    """Extract local shard from distributed array.

    For JAX distributed arrays, extracts the first addressable shard.
    Returns JAX Array which can be converted by the backend as needed.
    """
    if hasattr(value, "addressable_shards"):
        shards = value.addressable_shards
        if not shards:
            raise ValueError("No addressable shards found in distributed array.")
        return shards[0].data
    return value


__all__ = [
    "as_native_zerocopy",
    "extract_local_array",
    "to_cupy_array",
    "to_dlpack",
    "to_host_array",
    "to_jax_array",
    "to_jax_tree",
    "to_numpy",
]
