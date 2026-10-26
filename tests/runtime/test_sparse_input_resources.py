from fluxway.common.resource_registry import ResourceRegistry, bind_resource_registry
from sparse_embedding.checkpoint import PartitionSpec

from flare.runtime.fluxway.workers.sparse_input import (
    InputSPMD,
    get_input_spmd_class,
)


def test_input_registers_one_owner_for_shared_sparse_base() -> None:
    sparse_groups = ("hashed_sparse_dim_4", "hashed_sparse_dim_8")
    registry = ResourceRegistry()

    with bind_resource_registry(registry):
        InputSPMD(
            sparse_groups=sparse_groups,
            InputNodeOutput=object,
            key_dtype="int64",
            sparse_partition_assignment_spec=PartitionSpec.hash(),
        )

    strategy = registry.resolve("sparse_embedding", "hashed_sparse.assignment")
    assert callable(strategy)


def test_input_dynamic_routing_uses_registered_assignment_owner() -> None:
    bound_input = get_input_spmd_class(
        sparse_groups=("hashed_sparse_dim_4", "hashed_sparse_dim_8"),
        runtime_env=None,
        resource_profile=None,
        batch_shardings={"samples_valid_count": {"type": "static", "axis": []}},
        key_dtype="int64",
        sparse_partition_assignment_spec=PartitionSpec.hash(),
    )

    output_sharding = bound_input.process._fluxway_operator_spec.output_sharding
    strategy = output_sharding["unique_hashed_sparse_dim_4"].routing.strategy
    assert strategy.target.namespace == "sparse_embedding"
    assert strategy.target.resource_id == "hashed_sparse.assignment"
