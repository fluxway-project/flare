"""Unified JAX physical mesh materialization and device mesh container.

This module provides the canonical bridge from Fluxway DeviceMesh to JAX Mesh,
and the pure ParallelMesh physical device coordinates container.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import jax
import numpy as np
from fluxway.common.mesh import DeviceMesh
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def materialize_jax_mesh(
    device_mesh: DeviceMesh,
    *,
    device_id_to_jax: Mapping[int, Any] | None = None,
) -> Mesh:
    """Convert a Fluxway DeviceMesh to a jax.sharding.Mesh.

    This is the ONE canonical bridge where Fluxway topology meets JAX devices.
    Both training and inference use this exact function.
    """
    if not isinstance(device_mesh, DeviceMesh):
        raise TypeError("device_mesh must be a Fluxway DeviceMesh")

    axis_names = tuple(device_mesh.axis_names)
    shape = tuple(device_mesh.shape)
    if not axis_names or any(
        not isinstance(name, str) or not name for name in axis_names
    ):
        raise ValueError("axis_names must contain non-empty strings")
    if len(axis_names) != len(set(axis_names)):
        raise ValueError(f"axis_names must be unique, got {axis_names}")
    if len(axis_names) != len(shape):
        raise ValueError("axis_names and shape must have the same rank")
    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
        for size in shape
    ):
        raise ValueError(f"shape must contain positive integers, got {shape}")
    shape = tuple(int(size) for size in shape)
    raw_device_ids = tuple(device_mesh.flatten())
    if any(
        isinstance(device_id, bool) or not isinstance(device_id, Integral)
        for device_id in raw_device_ids
    ):
        raise TypeError("device IDs must be integer values")
    device_ids = tuple(int(device_id) for device_id in raw_device_ids)

    if math.prod(shape) != len(device_ids):
        raise ValueError(
            f"device count {len(device_ids)} does not match shape {shape} product {math.prod(shape)}"
        )

    int32_max = np.iinfo(np.int32).max
    for dev_id in device_ids:
        if dev_id < 0 or dev_id > int32_max:
            raise ValueError(f"device ID {dev_id} out of int32 bounds [0, {int32_max}]")

    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"device IDs must be unique, got {device_ids}")

    if device_id_to_jax is None:
        if device_ids != (0,) or jax.process_count() != 1:
            raise RuntimeError(
                "device_id_to_jax is required for multiple or non-local runtime ranks"
            )
        local_devices = tuple(jax.local_devices())
        if len(local_devices) != 1:
            raise RuntimeError(
                "device_id_to_jax is required when the process has multiple devices"
            )
        selected = list(local_devices)
    else:
        if not isinstance(device_id_to_jax, Mapping):
            raise TypeError("device_id_to_jax must be a mapping")
        missing = tuple(
            device_id for device_id in device_ids if device_id not in device_id_to_jax
        )
        if missing:
            raise KeyError(f"device_id_to_jax is missing device IDs: {missing}")
        selected = [device_id_to_jax[i] for i in device_ids]

    if any(not isinstance(device, jax.Device) for device in selected):
        raise TypeError("device_id_to_jax must contain JAX devices")
    if len(set(selected)) != len(selected):
        raise ValueError("device_id_to_jax must assign distinct JAX devices to ranks")

    device_array = np.asarray(selected, dtype=object).reshape(shape)
    return Mesh(device_array, axis_names=axis_names)


def _is_scalar(leaf: Any) -> bool:
    shape = getattr(leaf, "shape", None)
    if shape is not None:
        return len(shape) == 0
    return isinstance(leaf, (int, float, bool, np.number))


def state_with_sharding(value: Any, sharding: NamedSharding) -> Any:
    """Recursively assign sharding to abstract and concrete pytree leaves.

    Scalar leaves (such as counters, learning rates, or loss values) are
    automatically replicated via P() to comply with XLA device_put invariants.
    """
    if not isinstance(sharding, NamedSharding):
        raise TypeError(
            f"sharding must be jax.sharding.NamedSharding, got {type(sharding).__name__}"
        )
    replicated = NamedSharding(sharding.mesh, P()) if sharding.spec != P() else sharding

    def convert(leaf: Any) -> Any:
        target = replicated if _is_scalar(leaf) else sharding
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=target)
        return jax.device_put(leaf, target)

    return jax.tree_util.tree_map(convert, value)


@dataclass(frozen=True, slots=True)
class ParallelMesh:
    """Explicit multi-dimensional physical device mesh container.

    Focuses strictly on physical device coordinates, named axes, and foundational
    NamedSharding primitives. Zero coupling with training algorithms or ZeRO stages.
    """

    mesh: Mesh
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, Mesh):
            raise TypeError(
                f"mesh must be jax.sharding.Mesh, got {type(self.mesh).__name__}"
            )
        if not self.axis_names:
            raise ValueError("axis_names must be non-empty")
        if len(self.axis_names) != len(set(self.axis_names)):
            raise ValueError(f"axis_names must be unique, got {self.axis_names}")
        mesh_axes = tuple(self.mesh.axis_names)
        if mesh_axes != self.axis_names:
            raise ValueError(
                f"mesh axis names {mesh_axes} do not match declared axis_names {self.axis_names}"
            )

    @classmethod
    def from_device_mesh(
        cls,
        device_mesh: DeviceMesh,
        *,
        device_id_to_jax: Mapping[int, Any] | None = None,
    ) -> ParallelMesh:
        """Construct a ParallelMesh directly from a Fluxway DeviceMesh."""
        jax_mesh = materialize_jax_mesh(device_mesh, device_id_to_jax=device_id_to_jax)
        return cls(mesh=jax_mesh, axis_names=tuple(jax_mesh.axis_names))

    @classmethod
    def from_axes(
        cls,
        *,
        devices: Sequence[Any] | None = None,
        **axes: int,
    ) -> ParallelMesh:
        """Construct a validated multi-dimensional physical device mesh."""
        if not axes:
            raise ValueError("at least one axis must be specified")
        for name, size in axes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("axis name must be non-empty")
            if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
                raise ValueError(f"axis {name!r} size must be positive, got {size}")
        resolved_devices = tuple(jax.devices() if devices is None else devices)
        if not resolved_devices:
            raise ValueError("devices must not be empty")
        expected_size = math.prod(axes.values())
        if len(resolved_devices) != expected_size:
            raise ValueError(
                f"device count {len(resolved_devices)} does not match axes product {expected_size} "
                f"(axes={axes})"
            )
        shape = tuple(axes.values())
        axis_names = tuple(axes.keys())
        device_array = np.asarray(resolved_devices, dtype=object).reshape(shape)
        mesh = Mesh(device_array, axis_names)
        return cls(mesh=mesh, axis_names=axis_names)

    @classmethod
    def from_dp(
        cls,
        *,
        size: int | None = None,
        devices: Sequence[Any] | None = None,
        axis_name: str = "data",
    ) -> ParallelMesh:
        """Construct a 1D data-parallel mesh."""
        if not axis_name:
            raise ValueError("axis_name must be non-empty")
        resolved_devices = tuple(jax.devices() if devices is None else devices)
        if size is not None:
            if size <= 0:
                raise ValueError(f"size must be positive, got {size}")
            if size > len(resolved_devices):
                raise ValueError(
                    f"size {size} exceeds available device count {len(resolved_devices)}"
                )
            resolved_devices = resolved_devices[:size]
        if not resolved_devices:
            raise ValueError("devices must not be empty")
        return cls.from_axes(
            devices=resolved_devices, **{axis_name: len(resolved_devices)}
        )

    @property
    def devices(self) -> tuple[Any, ...]:
        return tuple(self.mesh.devices.flat)

    @property
    def size(self) -> int:
        return int(self.mesh.size)

    @property
    def shape(self) -> Mapping[str, int]:
        return self.mesh.shape

    @property
    def local_devices(self) -> tuple[Any, ...]:
        return tuple(self.mesh.local_devices)

    @property
    def replicated(self) -> NamedSharding:
        """Replicated sharding for step counters, scalars, and global broadcast state."""
        return NamedSharding(self.mesh, P())

    def named(self, *spec_items: Any) -> NamedSharding:
        """Construct arbitrary NamedSharding on this mesh."""
        return NamedSharding(self.mesh, P(*spec_items))

    def apply(self, value: Any, sharding: NamedSharding) -> Any:
        """Assign sharding to an abstract or concrete pytree."""
        return state_with_sharding(value, sharding)

    def axis_size(self, axis_name: str) -> int:
        if axis_name not in self.axis_names:
            raise KeyError(
                f"axis {axis_name!r} is not present in mesh axes {self.axis_names}"
            )
        return int(self.mesh.shape[axis_name])

    def __getitem__(self, axis_name: str) -> int:
        return self.axis_size(axis_name)


__all__ = [
    "ParallelMesh",
    "materialize_jax_mesh",
    "state_with_sharding",
]
