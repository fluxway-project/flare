from pathlib import Path

import jax
import numpy as np
import pytest
from fluxway.common.mesh import DeviceMesh
from sparse_embedding.checkpoint import (
    PartitionSpec,
    RestorePlan,
    encode_restore_options,
)

from flare.inference.weights import (
    LanceSparseWeightAdapter,
    OrbaxModelWeightAdapter,
    PublishedAssetEvidence,
    SparsePartitionMapping,
    bind_inference_weights,
)
from flare.jax.mesh import materialize_jax_mesh


class _ModelDefinition:
    def __init__(self, sparse_groups: tuple[str, ...] = ("boss",)) -> None:
        self.sparse_groups = sparse_groups

    def abstract_model_state(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"kernel": jax.ShapeDtypeStruct((1,), np.float32)}

    def validate_state(self, state: object) -> None:
        pass


def _asset(tmp_path: Path, *, groups: tuple[str, ...] = ("boss",)) -> Path:
    asset = tmp_path / "weights" / "steps" / "73"
    (asset / "model").mkdir(parents=True)
    for group in groups:
        (asset / "sparse" / group).mkdir(parents=True)
    return asset


def _device_mesh() -> DeviceMesh:
    device_count = len(jax.devices())
    return DeviceMesh(
        axis_names=("dp",),
        device_shape=(device_count,),
        device_ids=tuple(range(device_count)),
    )


def test_sparse_partition_mapping_from_device_mesh() -> None:
    device_mesh = DeviceMesh(
        axis_names=("dp",),
        device_shape=(4,),
        device_ids=(7, 3, 11, 5),
    )
    mapping = SparsePartitionMapping.from_device_mesh(device_mesh, sparse_axes=("dp",))
    assert mapping.sparse_axes == ("dp",)
    assert mapping.replica_axes == ()
    assert mapping.partition_count == 4


def test_sparse_partition_mapping_projects_ordered_axes_and_replica_axes() -> None:
    device_mesh = DeviceMesh(
        axis_names=("dp", "fsdp", "replica"),
        device_shape=(2, 3, 4),
        device_ids=tuple(range(24)),
    )
    mapping = SparsePartitionMapping.from_device_mesh(
        device_mesh, sparse_axes=("fsdp", "dp")
    )
    assert mapping.sparse_axes == ("fsdp", "dp")
    assert mapping.replica_axes == ("replica",)
    assert mapping.partition_count == 6
    mapping.validate_device_mesh(device_mesh)


def test_sparse_partition_mapping_rejects_string_axis_argument() -> None:
    with pytest.raises(TypeError, match="sequence of axis names"):
        SparsePartitionMapping.from_device_mesh(
            DeviceMesh(axis_names=("dp",), device_shape=(2,)), sparse_axes="dp"
        )


def test_sparse_partition_mapping_validations() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        SparsePartitionMapping((), ("dp",), (1,))
    with pytest.raises(ValueError, match="unique"):
        SparsePartitionMapping(("dp", "dp"), ("dp",), (1,))
    with pytest.raises(ValueError, match="present"):
        SparsePartitionMapping(("missing",), ("dp",), (1,))


def test_sparse_mapping_requires_explicit_valid_axes() -> None:
    with pytest.raises(TypeError, match="Fluxway DeviceMesh"):
        SparsePartitionMapping.from_device_mesh(object(), sparse_axes=("dp",))
    with pytest.raises(ValueError, match="present"):
        SparsePartitionMapping.from_device_mesh(
            DeviceMesh(axis_names=("dp", "tp"), device_shape=(2, 2)),
            sparse_axes=("missing",),
        )


def test_binding_pins_topology_and_rejects_unrelated_mapping(tmp_path: Path) -> None:
    device_mesh = DeviceMesh(axis_names=("dp",), device_ids=(7, 3))
    mapping = SparsePartitionMapping.from_device_mesh(device_mesh, sparse_axes=("dp",))
    with pytest.raises(ValueError, match="present"):
        bind_inference_weights(
            _asset(tmp_path),
            _ModelDefinition(),
            device_mesh,
            partition_mappings={"boss": SparsePartitionMapping(("tp",), ("dp",), (2,))},
        )
    source = bind_inference_weights(
        tmp_path / "weights" / "steps" / "73",
        _ModelDefinition(),
        device_mesh,
        partition_mappings={"boss": mapping},
    )
    assert tuple(source.device_mesh.flatten()) == (7, 3)
    assert source.device_mesh.is_frozen
    assert tuple(source.restore_payloads_by_partition["boss"]) == (0, 1)


def test_jax_materialization_requires_explicit_rank_device_binding() -> None:
    device_mesh = DeviceMesh(axis_names=("dp",), device_ids=(7, 3))
    with pytest.raises(RuntimeError, match="device_id_to_jax is required"):
        materialize_jax_mesh(device_mesh)
    with pytest.raises(KeyError, match="missing device IDs"):
        materialize_jax_mesh(device_mesh, device_id_to_jax={7: jax.devices()[0]})
    with pytest.raises(ValueError, match="distinct JAX devices"):
        materialize_jax_mesh(
            device_mesh, device_id_to_jax={7: jax.devices()[0], 3: jax.devices()[0]}
        )
    with pytest.raises(TypeError, match="JAX devices"):
        materialize_jax_mesh(device_mesh, device_id_to_jax={7: object(), 3: object()})
    devices = jax.devices()
    if len(devices) >= 2:
        materialized = materialize_jax_mesh(
            device_mesh, device_id_to_jax={7: devices[1], 3: devices[0]}
        )
        assert tuple(materialized.devices.flat) == (devices[1], devices[0])
    else:
        single_mesh = DeviceMesh(axis_names=("dp",), device_ids=(7,))
        materialized = materialize_jax_mesh(
            single_mesh, device_id_to_jax={7: devices[0]}
        )
        assert tuple(materialized.devices.flat) == (devices[0],)


def test_materialize_jax_mesh_from_device_mesh() -> None:
    device_mesh = DeviceMesh(
        axis_names=("dp",),
        device_shape=(1,),
        device_ids=(0,),
    )
    jm = materialize_jax_mesh(device_mesh, device_id_to_jax={0: jax.devices()[0]})
    assert jm.axis_names == ("dp",)
    assert jm.shape["dp"] == 1


def test_materialize_jax_mesh_rejects_lookalike_mesh() -> None:
    class DummyMesh:
        axis_names = ("dp",)
        shape = (1,)

        def flatten(self):
            return [2**31]

    with pytest.raises(TypeError, match="Fluxway DeviceMesh"):
        materialize_jax_mesh(DummyMesh())


def test_bind_inference_weights_requires_exact_published_layout(tmp_path: Path) -> None:
    asset = _asset(tmp_path, groups=("boss", "job"))
    source = bind_inference_weights(
        asset,
        _ModelDefinition(("boss", "job")),
        _device_mesh(),
        partition_mappings={
            "boss": SparsePartitionMapping.from_device_mesh(
                _device_mesh(), sparse_axes=("dp",)
            ),
            "job": SparsePartitionMapping.from_device_mesh(
                _device_mesh(), sparse_axes=("dp",)
            ),
        },
    )
    assert set(source.sparse) == {"boss", "job"}
    assert source.model.checkpoint_path == asset / "model"
    assert source.device_mesh.axis_names == ("dp",)
    assert set(source.partition_mappings) == {"boss", "job"}
    assert all(
        isinstance(adapter, LanceSparseWeightAdapter)
        for adapter in source.sparse.values()
    )


def test_bind_inference_weights_keeps_sparse_mesh_independent_of_dense_mesh(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path)
    dense_mesh = DeviceMesh(axis_names=("dp",), device_ids=(7, 3))
    sparse_mapping = SparsePartitionMapping.from_device_mesh(
        DeviceMesh(axis_names=("shard",), device_ids=(3, 7)),
        sparse_axes=("shard",),
    )

    source = bind_inference_weights(
        asset,
        _ModelDefinition(),
        dense_mesh,
        partition_mappings={"boss": sparse_mapping},
    )

    assert source.device_mesh.axis_names == ("dp",)
    assert source.partition_mappings["boss"].mesh_axis_names == ("shard",)
    assert source.partition_mappings["boss"].partition_count == 2


def test_bound_asset_rejects_internal_generation_change(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    source = bind_inference_weights(
        asset,
        _ModelDefinition(),
        _device_mesh(),
        partition_mappings={
            "boss": SparsePartitionMapping.from_device_mesh(
                _device_mesh(), sparse_axes=("dp",)
            )
        },
    )
    (asset / "model" / "replacement").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="contents changed"):
        source.asset_evidence.verify()


def test_bind_inference_weights_requires_explicit_mesh(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    with pytest.raises(TypeError, match="DeviceMesh"):
        bind_inference_weights(
            asset, _ModelDefinition(), None, partition_mappings={"boss": object()}
        )  # type: ignore[arg-type]


def test_bind_inference_weights_rejects_jax_mesh_inference(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    with pytest.raises(TypeError, match="DeviceMesh"):
        bind_inference_weights(
            asset,
            _ModelDefinition(),
            materialize_jax_mesh(
                _device_mesh(), device_id_to_jax=dict(enumerate(jax.devices()))
            ),  # type: ignore[arg-type]
            partition_mappings={"boss": object()},
        )


@pytest.mark.parametrize(
    "relative_path",
    ("weights", "weights/refs/steps/73", "weights/descriptor.msgpack"),
)
def test_bind_inference_weights_rejects_non_asset_paths(
    tmp_path: Path, relative_path: str
) -> None:
    path = tmp_path / relative_path
    if path.suffix:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    else:
        path.mkdir(parents=True)
    with pytest.raises(ValueError, match="inference weights must be an explicit"):
        bind_inference_weights(
            path,
            _ModelDefinition(),
            _device_mesh(),
            partition_mappings={
                "boss": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                )
            },
        )


def test_bind_inference_weights_rejects_missing_sparse_groups(tmp_path: Path) -> None:
    asset = _asset(tmp_path, groups=("boss",))
    with pytest.raises(ValueError, match="missing.*job"):
        bind_inference_weights(
            asset,
            _ModelDefinition(("boss", "job")),
            _device_mesh(),
            partition_mappings={
                "boss": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                ),
                "job": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                ),
            },
        )


def test_bind_inference_weights_rejects_unknown_sparse_groups(tmp_path: Path) -> None:
    asset = _asset(tmp_path, groups=("boss", "extra"))
    with pytest.raises(ValueError, match="unknown.*extra"):
        bind_inference_weights(
            asset,
            _ModelDefinition(("boss",)),
            _device_mesh(),
            partition_mappings={
                "boss": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                )
            },
        )


def test_bind_inference_weights_rejects_symlink_components(tmp_path: Path) -> None:
    root = tmp_path / "published"
    real_asset = root / "steps" / "10"
    (real_asset / "model").mkdir(parents=True)
    (real_asset / "sparse" / "boss").mkdir(parents=True)

    escaped = tmp_path / "outside"
    escaped.mkdir()
    symlink_step = root / "steps" / "11"
    symlink_step.symlink_to(escaped, target_is_directory=True)

    with pytest.raises(
        ValueError, match="published step directory must not be a symlink"
    ):
        bind_inference_weights(
            symlink_step,
            _ModelDefinition(),
            _device_mesh(),
            partition_mappings={
                "boss": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                )
            },
        )


def test_bind_inference_weights_rejects_published_latest_alias(tmp_path: Path) -> None:
    root = tmp_path / "published"
    real_asset = root / "steps" / "12"
    (real_asset / "model").mkdir(parents=True)
    (real_asset / "sparse" / "boss").mkdir(parents=True)

    latest = root / "latest"
    latest.symlink_to(Path("steps/12"))

    with pytest.raises(ValueError, match="immutable steps"):
        bind_inference_weights(
            latest,
            _ModelDefinition(),
            _device_mesh(),
            partition_mappings={
                "boss": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                )
            },
        )


def test_bind_inference_weights_rejects_step_internal_symlink(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    escaped = tmp_path / "outside"
    escaped.mkdir()
    symlink_child = asset / "model" / "child"
    symlink_child.symlink_to(escaped, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink descendant"):
        bind_inference_weights(
            asset,
            _ModelDefinition(),
            _device_mesh(),
            partition_mappings={
                "boss": SparsePartitionMapping.from_device_mesh(
                    _device_mesh(), sparse_axes=("dp",)
                )
            },
        )


def test_model_weight_adapter_rejects_modified_asset_contents(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    evidence = PublishedAssetEvidence.capture(asset, asset, None)
    adapter = OrbaxModelWeightAdapter(asset / "model", evidence)
    (asset / "model" / "tampered").write_bytes(b"bad")

    with pytest.raises(RuntimeError, match="contents changed"):
        adapter.load(
            _ModelDefinition(),
            materialize_jax_mesh(
                _device_mesh(), device_id_to_jax=dict(enumerate(jax.devices()))
            ),
        )


def test_sparse_adapter_requires_restore_payloads(tmp_path: Path) -> None:
    evidence = PublishedAssetEvidence.capture(tmp_path, tmp_path, None)
    adapter = LanceSparseWeightAdapter(tmp_path, evidence)
    with pytest.raises(TypeError, match="restore payload"):
        adapter.load(object(), b"")


def test_bind_inference_weights_generates_restore_payloads(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    device_mesh = DeviceMesh(
        axis_names=("dp",),
        device_shape=(2,),
        device_ids=(7, 3),
    )
    partition_mapping = SparsePartitionMapping.from_device_mesh(
        device_mesh, sparse_axes=("dp",)
    )
    assignment = PartitionSpec.modulo()
    source = bind_inference_weights(
        asset,
        _ModelDefinition(),
        device_mesh,
        partition_mappings={"boss": partition_mapping},
        sparse_partition_assignment_spec=assignment,
        key_dtype="int32",
    )
    assert source.restore_payloads_by_partition["boss"] == {
        0: encode_restore_options(
            RestorePlan.from_partition(
                spec=assignment,
                partition_count=2,
                partition_id=0,
                key_dtype="int32",
            )
        ),
        1: encode_restore_options(
            RestorePlan.from_partition(
                spec=assignment,
                partition_count=2,
                partition_id=1,
                key_dtype="int32",
            )
        ),
    }


def test_bind_inference_weights_defaults_to_owner_modulo_int64(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    device_mesh = DeviceMesh(
        axis_names=("dp",),
        device_shape=(1,),
        device_ids=(11,),
    )
    partition_mapping = SparsePartitionMapping.from_device_mesh(
        device_mesh, sparse_axes=("dp",)
    )
    source = bind_inference_weights(
        asset,
        _ModelDefinition(),
        device_mesh,
        partition_mappings={"boss": partition_mapping},
    )
    expected = encode_restore_options(
        RestorePlan.from_partition(
            spec=PartitionSpec.modulo(),
            partition_count=1,
            partition_id=0,
            key_dtype="int64",
        )
    )
    assert source.restore_payloads_by_partition["boss"] == {0: expected}
