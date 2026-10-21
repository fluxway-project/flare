from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral

from fluxway.common.mesh import DeviceMesh


@dataclass(frozen=True, slots=True)
class SparsePartitionMapping:
    """Logical sparse partition projection over explicit Fluxway mesh axes."""

    sparse_axes: tuple[str, ...]
    mesh_axis_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.sparse_axes)
        names = tuple(self.mesh_axis_names)
        shape = tuple(self.mesh_shape)
        if not axes:
            raise ValueError("sparse_axes must contain at least one axis")
        if len(names) != len(shape):
            raise ValueError("mesh_axis_names and mesh_shape must have the same rank")
        if len(names) != len(set(names)):
            raise ValueError("mesh axis names must be unique")
        if len(axes) != len(set(axes)):
            raise ValueError("sparse_axes must be unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("mesh axis names must be non-empty strings")
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("sparse_axes must contain non-empty strings")
        if any(axis not in names for axis in axes):
            raise ValueError(
                f"sparse_axes {axes!r} must be present in mesh axes {names!r}"
            )
        if any(
            isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
            for size in shape
        ):
            raise ValueError("mesh_shape must contain positive integers")
        object.__setattr__(self, "sparse_axes", axes)
        object.__setattr__(self, "mesh_axis_names", names)
        object.__setattr__(self, "mesh_shape", tuple(int(size) for size in shape))

    @classmethod
    def from_device_mesh(
        cls,
        device_mesh: DeviceMesh,
        *,
        sparse_axes: tuple[str, ...] | list[str],
    ) -> SparsePartitionMapping:
        if not isinstance(device_mesh, DeviceMesh):
            raise TypeError("device_mesh must be a Fluxway DeviceMesh")
        if isinstance(sparse_axes, str):
            raise TypeError("sparse_axes must be a sequence of axis names")
        return cls(
            sparse_axes=tuple(sparse_axes),
            mesh_axis_names=tuple(device_mesh.axis_names),
            mesh_shape=tuple(device_mesh.shape),
        )

    def validate_device_mesh(self, device_mesh: DeviceMesh) -> None:
        expected = self.from_device_mesh(device_mesh, sparse_axes=self.sparse_axes)
        if expected != self:
            raise ValueError("sparse partition mapping does not match the DeviceMesh")

    @property
    def partition_count(self) -> int:
        sizes = [
            self.mesh_shape[self.mesh_axis_names.index(axis)]
            for axis in self.sparse_axes
        ]
        return prod(sizes)

    @property
    def replica_axes(self) -> tuple[str, ...]:
        return tuple(
            axis for axis in self.mesh_axis_names if axis not in self.sparse_axes
        )


__all__ = ["SparsePartitionMapping"]
