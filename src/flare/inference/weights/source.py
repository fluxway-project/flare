from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlparse

import jax
import orbax.checkpoint as ocp
from fluxway.common.mesh import DeviceMesh
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sparse_embedding.checkpoint import (
    PartitionSpec,
)

from flare.mesh import SparsePartitionMapping
from flare.runtime.sparse_precision import require_key_dtype
from flare.runtime.sparse_restore import build_sparse_restore_plans


class ModelWeightAdapter(Protocol):
    def load(self, state_definition: Any, mesh: Mesh) -> Any: ...


class SparseWeightAdapter(Protocol):
    def load(self, store: Any, restore_payload: bytes) -> None: ...


def _validate_sparse_restore_payloads(
    restore_payloads_by_partition: Mapping[Any, object],
) -> Mapping[int, bytes]:
    if not isinstance(restore_payloads_by_partition, Mapping):
        raise TypeError("restore payloads must be a logical-partition mapping")
    normalized: dict[int, bytes] = {}
    for partition_id, restore_payload in restore_payloads_by_partition.items():
        if (
            isinstance(partition_id, bool)
            or not isinstance(partition_id, int)
            or partition_id < 0
        ):
            raise TypeError("logical partition ids must be non-negative integers")
        if not isinstance(restore_payload, bytes) or not restore_payload:
            raise TypeError("restore payloads must be non-empty opaque bytes")
        normalized[partition_id] = restore_payload
    if not normalized:
        raise ValueError("restore payloads must contain at least one logical partition")
    return MappingProxyType(normalized)


def select_sparse_restore_payload(
    restore_payloads_by_partition: Mapping[Any, object], logical_partition_id: int
) -> bytes:
    if isinstance(logical_partition_id, bool) or not isinstance(
        logical_partition_id, int
    ):
        raise TypeError("logical partition id must be an integer")
    normalized = _validate_sparse_restore_payloads(restore_payloads_by_partition)
    try:
        return normalized[logical_partition_id]
    except KeyError as error:
        raise KeyError(
            "composition did not provide a restore payload for logical partition "
            f"{logical_partition_id}"
        ) from error


@dataclass(frozen=True, slots=True)
class PublishedAssetEvidence:
    """POSIX evidence pinning one published asset binding."""

    requested_path: Path
    resolved_path: Path
    latest_target: str | None
    device: int
    inode: int
    mtime_ns: int
    tree_signature: tuple[tuple[str, int, int, int, int, int], ...]

    @staticmethod
    def _tree_signature(
        root: Path,
    ) -> tuple[tuple[str, int, int, int, int, int], ...]:
        entries: list[tuple[str, int, int, int, int, int]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(
                    f"published asset contains a symlink descendant: {path}"
                )
            stat = path.stat()
            kind = 1 if path.is_dir() else 0
            entries.append(
                (
                    relative,
                    kind,
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
        return tuple(entries)

    @classmethod
    def capture(
        cls,
        requested_path: Path,
        resolved_path: Path,
        latest_target: str | None,
    ) -> PublishedAssetEvidence:
        stat = resolved_path.stat()
        return cls(
            requested_path=requested_path,
            resolved_path=resolved_path,
            latest_target=latest_target,
            device=stat.st_dev,
            inode=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            tree_signature=cls._tree_signature(resolved_path),
        )

    def verify(self) -> None:
        if self.latest_target is not None:
            if not self.requested_path.is_symlink():
                raise RuntimeError("published latest alias disappeared")
            if os.readlink(self.requested_path) != self.latest_target:
                raise RuntimeError("published latest generation changed")
            if self.requested_path.resolve() != self.resolved_path:
                raise RuntimeError("published latest target changed")
        elif self.requested_path.resolve() != self.resolved_path:
            raise RuntimeError("published step path changed")
        stat = self.resolved_path.stat()
        if (
            stat.st_dev,
            stat.st_ino,
            stat.st_mtime_ns,
        ) != (self.device, self.inode, self.mtime_ns):
            raise RuntimeError("published asset generation changed")
        if self._tree_signature(self.resolved_path) != self.tree_signature:
            raise RuntimeError("published asset contents changed")


def _absolute_posix_path(value: str | Path) -> Path:
    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"unsupported inference weight URI scheme: {parsed.scheme!r}")
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file URI host must be empty or localhost")
        raw = parsed.path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _reject_symlink_escape(published_root: Path, resolved_path: Path) -> None:
    try:
        resolved_path.relative_to(published_root.resolve())
    except ValueError as error:
        raise ValueError(
            "inference weight path resolves outside its published root: "
            f"{published_root}"
        ) from error


def _validate_asset_path(value: str | Path) -> PublishedAssetEvidence:
    requested = _absolute_posix_path(value)
    if not requested.exists():
        raise FileNotFoundError(f"inference weight path does not exist: {requested}")
    if not requested.is_dir():
        raise ValueError(
            "inference weights must be an explicit steps/<step_id> directory "
            "or a published latest alias"
        )

    if requested.name == "latest":
        raise ValueError(
            "inference weights require an immutable steps/<step_id> path; latest is training-only"
        )
    elif requested.parent.name == "steps" and requested.name.isdecimal():
        published_root = requested.parent.parent
        if published_root.name == "refs":
            raise ValueError(
                "inference weights must be an explicit steps/<step_id> directory "
                "or a published latest alias"
            )
        if published_root.is_symlink() or requested.parent.is_symlink():
            raise ValueError("published step path must not contain symlink components")
        if requested.is_symlink():
            raise ValueError("published step directory must not be a symlink")
        resolved = requested.resolve()
        _reject_symlink_escape(published_root, resolved)
        asset_path = resolved
        latest_target = None
    else:
        raise ValueError(
            "inference weights must be an explicit steps/<step_id> directory "
            "or a published latest alias"
        )

    if asset_path.is_symlink():
        raise ValueError("published step directory must not be a symlink")
    children = {entry.name for entry in asset_path.iterdir()}
    if children != {"model", "sparse"}:
        raise ValueError(
            "published inference asset must contain exactly model/ and sparse/; "
            f"found={sorted(children)!r}"
        )
    model_path = asset_path / "model"
    sparse_path = asset_path / "sparse"
    if not model_path.is_dir() or model_path.is_symlink():
        raise ValueError("published inference asset model/ must be a real directory")
    if not sparse_path.is_dir() or sparse_path.is_symlink():
        raise ValueError("published inference asset sparse/ must be a real directory")
    return PublishedAssetEvidence.capture(requested, asset_path, latest_target)


def _validate_group_names(groups: Collection[str]) -> frozenset[str]:
    original = tuple(groups)
    resolved = frozenset(original)
    if any(not isinstance(group, str) or not group for group in resolved):
        raise ValueError("required sparse group names must be non-empty strings")
    if len(resolved) != len(original):
        raise ValueError("required sparse group names must be unique")
    return resolved


def _resolve_assignment_spec(
    assignment_spec: PartitionSpec | None,
) -> PartitionSpec:
    if assignment_spec is None:
        return PartitionSpec.modulo()
    if not isinstance(assignment_spec, PartitionSpec):
        raise TypeError("sparse_partition_assignment_spec must be a PartitionSpec")
    return assignment_spec


def _encode_sparse_restore_payloads(
    partition_mapping: SparsePartitionMapping,
    assignment_spec: PartitionSpec,
    key_dtype: str,
) -> Mapping[int, bytes]:
    return build_sparse_restore_plans(
        partition_mappings={"sparse": partition_mapping},
        assignment_spec=assignment_spec,
        key_dtype=key_dtype,
    )["sparse"]


@dataclass(frozen=True, slots=True)
class InferenceWeights:
    """Path binding for model state and independent sparse groups."""

    model: ModelWeightAdapter
    sparse: Mapping[str, SparseWeightAdapter]
    device_mesh: DeviceMesh
    partition_mappings: Mapping[str, SparsePartitionMapping]
    restore_payloads_by_partition: Mapping[str, Mapping[int, bytes]]
    asset_evidence: PublishedAssetEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.device_mesh, DeviceMesh):
            raise TypeError("device_mesh must be a Fluxway DeviceMesh")
        if not isinstance(self.sparse, Mapping):
            raise TypeError("inference sparse adapters must be a mapping")
        groups = set(self.sparse)
        if any(not isinstance(name, str) or not name for name in groups):
            raise ValueError("inference sparse group names must be non-empty strings")
        if (
            set(self.partition_mappings) != groups
            or set(self.restore_payloads_by_partition) != groups
        ):
            raise ValueError(
                "each sparse group requires independent mapping and restore payloads"
            )
        mappings = {}
        restore_payloads = {}
        for group in groups:
            mapping = self.partition_mappings[group]
            if not isinstance(mapping, SparsePartitionMapping):
                raise TypeError(
                    f"partition mapping for {group!r} must be SparsePartitionMapping"
                )
            normalized = _validate_sparse_restore_payloads(
                self.restore_payloads_by_partition[group]
            )
            if set(normalized) != set(range(mapping.partition_count)):
                raise ValueError(
                    f"restore payloads for {group!r} must cover logical partitions"
                )
            mappings[group] = mapping
            restore_payloads[group] = normalized
        object.__setattr__(self, "device_mesh", self.device_mesh.frozen_copy())
        object.__setattr__(self, "sparse", MappingProxyType(dict(self.sparse)))
        object.__setattr__(self, "partition_mappings", MappingProxyType(mappings))
        object.__setattr__(
            self, "restore_payloads_by_partition", MappingProxyType(restore_payloads)
        )
        if not isinstance(self.asset_evidence, PublishedAssetEvidence):
            raise TypeError("asset_evidence must be PublishedAssetEvidence")

    @classmethod
    def bind_from_weights_path(
        cls,
        weights_path: str | Path,
        model_definition: Any,
        device_mesh: DeviceMesh,
        *,
        partition_mappings: Mapping[str, SparsePartitionMapping],
        sparse_partition_assignment_spec: PartitionSpec | None = None,
        key_dtype: str = "int64",
    ) -> InferenceWeights:
        if not hasattr(model_definition, "abstract_model_state") or not hasattr(
            model_definition, "sparse_groups"
        ):
            raise TypeError(
                "model_definition must expose abstract_model_state() and sparse_groups"
            )
        if not isinstance(device_mesh, DeviceMesh):
            raise TypeError("device_mesh must be a Fluxway DeviceMesh")
        required = _validate_group_names(model_definition.sparse_groups)
        if set(partition_mappings) != required:
            raise ValueError("partition_mappings must match model sparse groups")
        assignment_spec = _resolve_assignment_spec(sparse_partition_assignment_spec)
        key_dtype = require_key_dtype(key_dtype)
        evidence = _validate_asset_path(weights_path)
        asset_path = evidence.resolved_path
        sparse_root = asset_path / "sparse"
        actual = frozenset(entry.name for entry in sparse_root.iterdir())
        if actual != required:
            raise ValueError(
                f"published sparse groups do not match the model definition: missing={sorted(required - actual)!r}, unknown={sorted(actual - required)!r}"
            )
        sparse_adapters = {
            group: LanceSparseWeightAdapter(
                weight_path=sparse_root / group, asset_evidence=evidence
            )
            for group in sorted(required)
        }
        restore_payloads = {
            group: _encode_sparse_restore_payloads(
                partition_mappings[group], assignment_spec, key_dtype
            )
            for group in required
        }
        return cls(
            model=OrbaxModelWeightAdapter(
                checkpoint_path=asset_path / "model", asset_evidence=evidence
            ),
            sparse=sparse_adapters,
            device_mesh=device_mesh,
            partition_mappings=partition_mappings,
            restore_payloads_by_partition=restore_payloads,
            asset_evidence=evidence,
        )


def bind_inference_weights(
    weights_path: str | Path,
    model_definition: Any,
    device_mesh: DeviceMesh,
    *,
    partition_mappings: Mapping[str, SparsePartitionMapping],
    sparse_partition_assignment_spec: PartitionSpec | None = None,
    key_dtype: str = "int64",
) -> InferenceWeights:
    return InferenceWeights.bind_from_weights_path(
        weights_path,
        model_definition,
        device_mesh,
        partition_mappings=partition_mappings,
        sparse_partition_assignment_spec=sparse_partition_assignment_spec,
        key_dtype=key_dtype,
    )


@dataclass(frozen=True, slots=True)
class OrbaxModelWeightAdapter:
    checkpoint_path: Path
    asset_evidence: PublishedAssetEvidence

    def load(self, state_definition: Any, mesh: Mesh) -> Any:
        self.asset_evidence.verify()
        if not hasattr(state_definition, "abstract_model_state"):
            raise TypeError(
                "model weight loading requires a model-only state definition"
            )
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a jax.sharding.Mesh")
        abstract_state = state_definition.abstract_model_state()
        target_sharding = NamedSharding(mesh, P())
        target_state = jax.tree_util.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(
                leaf.shape, leaf.dtype, sharding=target_sharding
            ),
            abstract_state,
        )
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        try:
            restored = checkpointer.restore(
                self.checkpoint_path,
                args=ocp.args.StandardRestore(item=target_state),
            )
        finally:
            checkpointer.close()
        if isinstance(restored, Mapping) and set(restored) == {"model_state"}:
            raise ValueError("model asset must contain the model-only state directly")
        state_definition.validate_state(restored)
        return restored


@dataclass(frozen=True, slots=True)
class LanceSparseWeightAdapter:
    weight_path: Path
    asset_evidence: PublishedAssetEvidence

    def load(self, store: Any, restore_payload: bytes) -> None:
        # pyrefly: ignore[missing-import]
        from sparse_embedding.checkpoint import LocalLanceStorage

        if not isinstance(restore_payload, bytes) or not restore_payload:
            raise TypeError("sparse restore requires explicit restore payload")
        self.asset_evidence.verify()

        storage = LocalLanceStorage.from_published_weight(self.weight_path)
        try:
            storage.load_inference_weights(store, restore_payload)
        finally:
            storage.close()


__all__ = [
    "InferenceWeights",
    "LanceSparseWeightAdapter",
    "ModelWeightAdapter",
    "OrbaxModelWeightAdapter",
    "PublishedAssetEvidence",
    "SparsePartitionMapping",
    "SparseWeightAdapter",
    "bind_inference_weights",
    "select_sparse_restore_payload",
]
