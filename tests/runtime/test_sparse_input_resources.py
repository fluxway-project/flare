import asyncio

import numpy as np
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.resource_registry import ResourceRegistry, bind_resource_registry
from fluxway.common.types.tensor import TransportTensor
from sparse_embedding.checkpoint import PartitionSpec

from flare.ops.sparse_unique import unique_first_occurrence
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


def test_input_cpu_deduplication_preserves_routing_contract() -> None:
    class Output:
        def __init__(self, **values):
            self.__dict__.update(values)

    sparse = TransportTensor(
        flat_values=np.asarray([1, 2, 1, 3], dtype=np.int64),
        nested_splits=[np.asarray([0, 2, 4], dtype=np.int32)],
        origin_indices=ExplicitIndex(np.asarray([0, 1], dtype=np.int32)),
        global_shape=(2, 2),
    )

    async def run():
        with bind_resource_registry(ResourceRegistry()):
            worker = InputSPMD(
                sparse_groups=("group_dim_2",),
                InputNodeOutput=Output,
                key_dtype="int64",
                sparse_partition_assignment_spec=PartitionSpec.hash(),
            )
            worker.rank = 0
            worker.world_size = 1
            worker.ctx = type("Context", (), {"step": 0})()
            return await worker.process(group_dim_2=sparse)

    result = asyncio.run(run())
    assert result.unique_group_dim_2.flat_values.tolist() == [1, 2, 3]
    assert result.inv_group_dim_2.flat_values.tolist() == [0, 1, 0, 2]
    assert result.unique_group_dim_2.origin_indices_materialized.tolist() == [0, 1, 3]


def test_cpu_unique_contract_matches_first_occurrence_semantics() -> None:
    from sparse_embedding import async_unique

    cases = (
        np.asarray([], dtype=np.int64),
        np.asarray([7, 7, 7], dtype=np.int64),
        np.asarray([3, 1, 3, 2, 1], dtype=np.int64),
        np.asarray([-1, 0, -1, np.iinfo(np.int64).min], dtype=np.int64),
        np.asarray([2**63, 0, 2**63, 2**64 - 1], dtype=np.uint64),
    )
    for values in cases:
        unique, inverse, first_indices = unique_first_occurrence(values)
        assert np.array_equal(unique[inverse], values)
        assert np.array_equal(values[first_indices], unique)
        assert inverse.dtype == np.int32
        assert first_indices.dtype == np.int64

        async def compare_native(keys):
            result = await async_unique(
                keys,
                return_inverse=True,
                return_first_indices=True,
            )
            return tuple(np.from_dlpack(item) for item in result)

        native_unique, native_inverse, native_first_indices = asyncio.run(
            compare_native(values)
        )
        assert np.array_equal(unique, native_unique)
        assert np.array_equal(inverse, native_inverse)
        assert np.array_equal(first_indices, native_first_indices)


def test_cpu_unique_rejects_non_integer_keys() -> None:
    with np.testing.assert_raises(TypeError):
        unique_first_occurrence(np.asarray([1.0, 2.0]))
