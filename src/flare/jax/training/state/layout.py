from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FlatStateLeafLayout:
    """Layout metadata for a single tensor leaf in a flattened, padded state representation.

    All multi-dimensional tensor leaves are flattened to 1D and right-padded so that their total
    element count becomes a multiple of mesh_size. This enables uniform sharding across devices
    for ZeRO-1/2/3 stages.
    """

    original_shape: tuple[int, ...]
    padded_size: int
    mesh_size: int
    dtype: str

    @classmethod
    def from_value(cls, value: Any, mesh_size: int) -> FlatStateLeafLayout:
        if mesh_size <= 0:
            raise ValueError(f"mesh_size must be positive, got {mesh_size}")
        original_shape = tuple(int(dimension) for dimension in value.shape)
        original_size = (
            int(np.prod(original_shape, dtype=np.int64)) if original_shape else 1
        )
        # Pad total element count to the nearest multiple of mesh_size for even sharding.
        padded_size = ((original_size + mesh_size - 1) // mesh_size) * mesh_size
        return cls(
            original_shape=original_shape,
            padded_size=padded_size,
            mesh_size=mesh_size,
            dtype=str(jnp.dtype(value.dtype)),
        )

    @classmethod
    def from_abstract(
        cls, value: jax.ShapeDtypeStruct, mesh_size: int
    ) -> FlatStateLeafLayout:
        if not isinstance(value, jax.ShapeDtypeStruct):
            raise TypeError(
                "FlatStateLeafLayout.from_abstract requires a JAX abstract leaf, "
                f"got {type(value).__name__}"
            )
        return cls.from_value(value, mesh_size)

    @property
    def original_size(self) -> int:
        if not self.original_shape:
            return 1
        return int(np.prod(self.original_shape, dtype=np.int64))

    @property
    def local_size(self) -> int:
        """Element count assigned to each device after partitioning."""
        return self.padded_size // self.mesh_size

    def to_padded_flat(
        self, value: Any, *, replicate_scalar_padding: bool = True
    ) -> Any:
        """Flatten tensor to 1D and right-pad with zeros (or replicated scalar) up to padded_size."""
        flat = jnp.reshape(jnp.asarray(value), (-1,))
        padding_size = self.padded_size - flat.shape[0]
        if padding_size == 0:
            return flat
        if replicate_scalar_padding and self.original_size == 1:
            padding = jnp.repeat(flat, padding_size)
        else:
            padding = jnp.zeros((padding_size,), dtype=flat.dtype)
        return jnp.concatenate((flat, padding))

    def from_padded_flat(self, value: Any) -> Any:
        """Slice off the padding elements and reshape back to the original multi-dimensional shape."""
        return jnp.reshape(value[: self.original_size], self.original_shape)

    def to_padded_gradient(self, value: Any) -> Any:
        """Flatten gradient tensor to 1D and right-pad with zeros for collective reduction."""
        return self.to_padded_flat(value, replicate_scalar_padding=False)


@dataclass(frozen=True)
class FlatStateDtypeBucketLayout:
    dtype: str
    leaves: tuple[FlatStateLeafLayout, ...]
    mesh_size: int

    @property
    def leaf_indices(self) -> tuple[int, ...]:
        return self._indices

    _indices: tuple[int, ...] = ()

    @property
    def local_size(self) -> int:
        return sum(leaf.local_size for leaf in self.leaves)


@dataclass(frozen=True)
class FlatStateLayout:
    leaves: tuple[FlatStateLeafLayout, ...]
    mesh_size: int
    buckets: tuple[FlatStateDtypeBucketLayout, ...] = ()

    @classmethod
    def from_values(cls, values: tuple[Any, ...], mesh_size: int) -> FlatStateLayout:
        return cls.from_leaves(
            tuple(FlatStateLeafLayout.from_value(v, mesh_size) for v in values),
            mesh_size,
        )

    @classmethod
    def from_abstract(
        cls, values: tuple[jax.ShapeDtypeStruct, ...], mesh_size: int
    ) -> FlatStateLayout:
        return cls.from_leaves(
            tuple(FlatStateLeafLayout.from_abstract(v, mesh_size) for v in values),
            mesh_size,
        )

    @classmethod
    def from_leaves(
        cls, leaves: tuple[FlatStateLeafLayout, ...], mesh_size: int
    ) -> FlatStateLayout:
        if mesh_size <= 0:
            raise ValueError(f"mesh_size must be positive, got {mesh_size}")
        indices_by_dtype: dict[str, list[int]] = {}
        for index, leaf in enumerate(leaves):
            indices_by_dtype.setdefault(leaf.dtype, []).append(index)
        buckets = tuple(
            FlatStateDtypeBucketLayout(
                dtype=dtype,
                leaves=tuple(leaves[index] for index in indices),
                mesh_size=mesh_size,
                _indices=tuple(indices),
            )
            for dtype, indices in indices_by_dtype.items()
        )
        return cls(leaves=leaves, mesh_size=mesh_size, buckets=buckets)

    @property
    def gradient_reduction_dtype(self) -> jnp.dtype:
        if not self.leaves:
            raise ValueError("gradient layout must contain at least one leaf")
        dtypes = tuple(jnp.dtype(leaf.dtype) for leaf in self.leaves)
        if any(not jnp.issubdtype(dtype, jnp.inexact) for dtype in dtypes):
            raise TypeError(
                "gradient reduction requires floating or complex leaves, "
                f"got {tuple(str(dtype) for dtype in dtypes)}"
            )
        with jax.numpy_dtype_promotion("standard"):
            return jnp.result_type(*dtypes)

    def prefix(self, count: int) -> FlatStateLayout:
        if count < 0 or count > len(self.leaves):
            raise ValueError(f"count must be in [0, {len(self.leaves)}], got {count}")
        return self.from_leaves(self.leaves[:count], self.mesh_size)

    def to_padded_leaves(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            leaf.to_padded_flat(val)
            for leaf, val in zip(self.leaves, values, strict=True)
        )

    def from_padded_leaves(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            leaf.from_padded_flat(val)
            for leaf, val in zip(self.leaves, values, strict=True)
        )

    def materialize_sharded_leaves(
        self, local_values: tuple[Any, ...], *, axis_name: str = "dp"
    ) -> tuple[Any, ...]:
        """All-gather sharded parameter slices and restore their original shapes (ZeRO-3)."""
        if len(local_values) != len(self.leaves):
            raise ValueError(
                f"materialize received {len(local_values)} values, expected {len(self.leaves)}"
            )
        if self.mesh_size == 1:
            return self.from_padded_leaves(local_values)
        materialized = []
        for value, leaf in zip(local_values, self.leaves, strict=True):
            # Gather all 1D local shards along the DP mesh axis into a complete padded vector.
            gathered = jax.lax.all_gather(
                value, axis_name=axis_name, axis=0, tiled=True
            )
            # Strip the padding and reshape back to original multi-dimensional tensor shape.
            materialized.append(leaf.from_padded_flat(jnp.reshape(gathered, (-1,))))
        return tuple(materialized)


__all__ = [
    "FlatStateDtypeBucketLayout",
    "FlatStateLayout",
    "FlatStateLeafLayout",
]
