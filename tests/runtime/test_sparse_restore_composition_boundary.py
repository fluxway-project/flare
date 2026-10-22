from pathlib import Path

import pytest
from sparse_embedding.checkpoint import PartitionSpec

from flare.inference.weights import select_sparse_restore_payload
from flare.mesh import (
    SparsePartitionMapping,
    build_1d_device_mesh,
    build_sparse_partition_mappings,
)
from flare.mesh import (
    bind_sparse_worker_meshes as bind_shared_sparse_worker_meshes,
)
from recipes.recsys.ranking.topology import (
    bind_sparse_checkpoint_restore_options,
    bind_sparse_embedding_restore_plans,
    bind_sparse_embedding_worker_restore_plans,
    bind_sparse_worker_meshes,
)


def test_training_checkpoint_options_use_logical_partition_encoder() -> None:
    options = bind_sparse_checkpoint_restore_options(
        partition_mappings={
            "wide": SparsePartitionMapping(("shard",), ("shard",), (2,)),
        },
        worker_counts={"wide": 2},
        assignment_spec=PartitionSpec.modulo(),
        key_dtype="int64",
    )
    assert set(options) == {"Embedding_wide"}
    assert set(options["Embedding_wide"]) == {0, 1}
    assert set(options["Embedding_wide"][0]) == {"wide_embedding"}
    assert options["Embedding_wide"][0]["wide_embedding"] != b""


ROOT = Path(__file__).parents[2]


def test_sparse_worker_consumes_composition_bound_restore_bytes() -> None:
    source = (
        ROOT / "src/flare/runtime/fluxway/workers/dynamic_embedding.py"
    ).read_text(encoding="utf-8")
    input_source = (
        ROOT / "src/flare/runtime/fluxway/workers/sparse_input.py"
    ).read_text(encoding="utf-8")

    assert "sparse_partition_assignment_spec" in source
    restore_source = (ROOT / "src/flare/runtime/sparse_restore.py").read_text(
        encoding="utf-8"
    )
    assert "RestorePlan.from_partition(" in restore_source
    assert "ReshardRestore(" not in restore_source
    assert "select_sparse_restore_payload" not in source
    assert "create_strategy" in source
    assert '"resource_id": f"{gname}.assignment"' in source
    assert 'spec["key_dtype"]' in source
    for source_text in (source, input_source):
        assert "RouterDescriptor" not in source_text
        assert "router_descriptor" not in source_text
        assert "normalize_router_keys" not in source_text
        assert "require_router_key_dtype" not in source_text
        assert "key_dtype" in source_text
    assert "sparse_adapter.restore_payload(" not in source
    assert "self.world_size,\n                self.rank" not in source


def test_recipe_passes_restore_plans_without_reconstructing_them() -> None:
    source = (ROOT / "recipes/recsys/ranking/graph/pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "restore_payloads_by_partition" in source
    assert "bind_sparse_embedding_worker_restore_plans(" in source
    assert "restore_plans_by_partition=restore_plans_by_partition" in source
    assert "sparse_partition_assignment_spec: PartitionSpec" in source
    assert '"key_dtype": common.key_dtype' in source
    assert "key_dtype=key_dtype" in source
    assert "router_descriptor" not in source
    assert "RouterDescriptor" not in source
    assert "sparse_adapter.restore_payload" not in source


def test_flare_does_not_import_private_sparse_checkpoint_apis() -> None:
    paths = [
        *((ROOT / "src/flare/inference").rglob("*.py")),
        *((ROOT / "src/flare/runtime/fluxway").rglob("*.py")),
        *((ROOT / "recipes/recsys").rglob("*.py")),
    ]
    forbidden = (
        "sparse_embedding._core.checkpoint",
        "import_batch_dlpack",
        "export_batches_dlpack",
        "SparseWeightSnapshot",
        "restore_weight_snapshot",
        "CheckpointRepository.latest",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(symbol in source for symbol in forbidden), path


def test_restore_payload_selection_validates_logical_partition_binding() -> None:
    assert (
        select_sparse_restore_payload({3: b"restore-payload"}, 3) == b"restore-payload"
    )
    with pytest.raises(KeyError, match="logical partition 2"):
        select_sparse_restore_payload({3: b"restore-payload"}, 2)
    with pytest.raises(TypeError, match="non-empty opaque bytes"):
        select_sparse_restore_payload({3: b""}, 3)
    with pytest.raises(ValueError, match="at least one logical partition"):
        select_sparse_restore_payload({}, 3)
    with pytest.raises(
        TypeError, match="logical partition ids must be non-negative integers"
    ):
        select_sparse_restore_payload({"3": b"restore-payload"}, 3)


def test_composition_helper_binds_complete_logical_partition_mapping() -> None:
    mapping = SparsePartitionMapping(("dp",), ("dp",), (2,))
    bound = bind_sparse_embedding_restore_plans(
        partition_mappings={"wide": mapping},
        restore_plans_by_partition={"wide": {0: b"first", 1: b"second"}},
    )
    assert bound == {"wide": {0: b"first", 1: b"second"}}
    with pytest.raises(TypeError):
        bound["wide"] = {}
    with pytest.raises(TypeError):
        bound["wide"][0] = b"changed"

    with pytest.raises(ValueError, match="logical partitions"):
        bind_sparse_embedding_restore_plans(
            partition_mappings={"wide": mapping},
            restore_plans_by_partition={"wide": {0: b"only"}},
        )


def test_composition_helper_binds_independent_group_worker_payloads() -> None:
    bindings = bind_sparse_embedding_worker_restore_plans(
        partition_mappings={
            "wide": SparsePartitionMapping(("dp",), ("dp", "replica"), (2, 2)),
            "deep": SparsePartitionMapping(("fsdp",), ("fsdp",), (3,)),
        },
        restore_plans_by_partition={
            "wide": {0: b"wide-0", 1: b"wide-1"},
            "deep": {0: b"deep-0", 1: b"deep-1", 2: b"deep-2"},
        },
    )
    assert bindings["wide"].axes == ("dp",)
    assert bindings["deep"].axes == ("fsdp",)
    assert bindings["wide"].values[1] == b"wide-1"
    assert bindings["deep"].values[2] == b"deep-2"
    with pytest.raises(ValueError, match="logical partitions"):
        bind_sparse_embedding_worker_restore_plans(
            partition_mappings={"wide": SparsePartitionMapping(("dp",), ("dp",), (2,))},
            restore_plans_by_partition={"wide": {0: b"only"}},
        )


def test_inference_sparse_meshes_are_validated_independently_per_group() -> None:
    mappings = {
        "wide": SparsePartitionMapping(("shard",), ("shard",), (2,)),
        "deep": SparsePartitionMapping(("shard",), ("shard",), (3,)),
    }

    meshes = bind_sparse_worker_meshes(
        partition_mappings=mappings,
        worker_counts={"wide": 2, "deep": 3},
    )

    assert meshes["wide"].axis_names == ("shard",)
    assert meshes["wide"].shape == (2,)
    assert meshes["deep"].shape == (3,)

    with pytest.raises(ValueError, match="does not match the DeviceMesh"):
        bind_sparse_worker_meshes(
            partition_mappings=mappings,
            worker_counts={"wide": 4, "deep": 3},
        )


def test_shared_mesh_helpers_keep_dense_and_sparse_topologies_independent() -> None:
    dense = build_1d_device_mesh(2, axis_name="dp")
    mappings = build_sparse_partition_mappings({"wide": 2, "deep": 3})
    sparse = bind_shared_sparse_worker_meshes(
        partition_mappings=mappings,
        worker_counts={"wide": 2, "deep": 3},
    )

    assert dense.axis_names == ("dp",)
    assert dense.shape == (2,)
    assert mappings["wide"].mesh_axis_names == ("shard",)
    assert mappings["deep"].partition_count == 3
    assert sparse["wide"].axis_names == ("shard",)
    assert sparse["deep"].shape == (3,)
