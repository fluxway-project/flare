from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from fluxway import PlacementRecipe, ResourceProfile
from fluxway.common.compiled_graph import SPMDGraphSpec, compile_graph
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.placement import ClusterNode, ClusterSnapshot, PlacementPlanner
from fluxway.placement.inventory import PlacementInventory

from flare.runtime.fluxway.placement import build_affinity_placement
from flare.runtime.fluxway.resources.resolution import resolve_resource_profiles


@dataclass(frozen=True)
class Ref:
    spmd_id: str
    spmd_class: type


def ref(name: str) -> Ref:
    return Ref(name, type(name, (), {}))


def test_generic_resource_resolver_uses_application_profiles() -> None:
    profiles = resolve_resource_profiles(
        {"producer": 2, "accelerator": 1, "driver": 1},
        {
            "producer": ResourceProfile(base={"CPU": 2}),
            "accelerator": ResourceProfile(base={"CPU": 1, "GPU": 1}),
        },
    )
    assert profiles.get_role("producer").base["CPU"] == Decimal(2)
    assert profiles.get_role("accelerator").base["GPU"] == Decimal(1)
    with pytest.raises(KeyError, match="no ResourceProfile"):
        resolve_resource_profiles({"unknown": 1}, {})


def test_fluxway_native_placement_recipe() -> None:
    graph, refs = _compiled_worker_graph(2, 4, 2, ((0, 2), (1, 3)))
    recipe = build_affinity_placement(
        recipe_id="native",
        compiled_graph=graph,
        reader_spmd=refs["Reader"],
        input_spmd=refs["Input"],
        sparse_table_spmds={"table_a": refs["TableA"], "table_b": refs["TableB"]},
        compute_spmd=refs["Compute"],
    )
    assert isinstance(recipe, PlacementRecipe)
    assert recipe.recipe_id == "native"
    assert recipe.constraints
    assert recipe.objective is not None


def test_fluxway_native_placement_excludes_requested_table() -> None:
    graph, refs = _compiled_worker_graph(2, 4, 2, ((0, 2), (1, 3)))
    recipe = build_affinity_placement(
        recipe_id="native",
        compiled_graph=graph,
        reader_spmd=refs["Reader"],
        input_spmd=refs["Input"],
        sparse_table_spmds={"table_a": refs["TableA"], "table_b": refs["TableB"]},
        compute_spmd=refs["Compute"],
        excluded_affinity_tables={"table_b"},
    )
    assert recipe.constraints


def _compiled_worker_graph(dp, readers, sparse, groups):
    profiles = resolve_resource_profiles(
        {
            "reader": readers,
            "input": dp,
            "compute": dp,
            "table": 5 * sparse,
            "metrics": 1,
        },
        {
            "reader": ResourceProfile(base={"CPU": 1}),
            "input": ResourceProfile(base={"CPU": 2}),
            "compute": ResourceProfile(base={"CPU": 2, "GPU": 1}),
            "table": ResourceProfile(base={"CPU": 1}),
            "metrics": ResourceProfile(base={"CPU": 1}),
        },
    )
    role_counts = {
        "Reader": ("reader", readers),
        "Input": ("input", dp),
        "Compute": ("compute", dp),
        "Metrics": ("metrics", 1),
        **{
            name: ("table", sparse)
            for name in ("TableA", "TableB", "TableC", "TableD", "TableE")
        },
    }
    refs = {name: ref(name) for name in role_counts}
    graph = compile_graph(
        name="reader-lane-placement",
        spmds=tuple(
            SPMDGraphSpec.create(
                spmd_id=name,
                spmd_class=refs[name].spmd_class,
                mesh=DeviceMesh(("data",), (count,)),
                resource_profile=profiles.get_role(role),
                operators={
                    "read": OperatorSpec(
                        name="read",
                        is_interleaved=True,
                        interleave_unit_size=len(groups[0]),
                        execution_groups=groups,
                    )
                }
                if name == "Reader"
                else {},
            )
            for name, (role, count) in sorted(role_counts.items())
        ),
        dependencies=(),
    )
    return graph, refs


def _worker_placement_recipe(graph, refs, policy):
    return build_affinity_placement(
        recipe_id="reader-lanes",
        compiled_graph=graph,
        reader_spmd=refs["Reader"],
        input_spmd=refs["Input"],
        sparse_table_spmds={
            "table_a": refs["TableA"],
            "table_b": refs["TableB"],
            "table_c": refs["TableC"],
            "table_d": refs["TableD"],
            "table_e": refs["TableE"],
        },
        compute_spmd=refs["Compute"],
        excluded_affinity_tables=frozenset({"table_d", "table_e"}),
        policy=policy,
    )


@pytest.mark.parametrize("policy", ("balanced", "feasible"))
@pytest.mark.parametrize("fraction", (Decimal("0.5"), Decimal("0.25")))
@pytest.mark.parametrize(
    "dp,readers,sparse,groups",
    (
        (2, 4, 2, ((0, 2), (1, 3))),
        (4, 8, 4, ((0, 2, 4, 6), (1, 3, 5, 7))),
        (2, 4, 2, ((1, 0), (2, 3))),
    ),
)
def test_recipe_colocates_compiled_reader_lanes(
    dp, readers, sparse, groups, fraction, policy
):
    graph, refs = _compiled_worker_graph(dp, readers, sparse, groups)
    total = {"CPU": Decimal(48), "GPU": Decimal(4), "memory": Decimal(103725028066)}
    cluster = ClusterSnapshot.from_nodes(
        ClusterNode(
            node_id=node,
            resources={key: value * share for key, value in total.items()},
        )
        for node, share in (("node-a", fraction), ("node-b", 1 - fraction))
    )
    recipe = _worker_placement_recipe(graph, refs, policy)
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        inventory=PlacementInventory.from_compiled_graph(graph),
        recipe=recipe,
        snapshot=cluster,
    )
    placement = {(item.spmd_id, item.rank): item.node_id for item in plan.assignments}
    execution = graph.spmd("Reader").operator("read").execution_layout
    for group in execution.groups:
        for lane in range(execution.logical_mesh.size):
            assert (
                placement[("Reader", group.physical_rank(lane))]
                == placement[("Input", lane)]
            )
    if policy == "feasible" and fraction == Decimal("0.5"):
        # Feasible placement must spread every worker class when the two
        # nodes have equal capacity, including tables excluded from the
        # affinity constraints.
        for spmd_id in (
            "Input",
            "Compute",
            "TableA",
            "TableB",
            "TableC",
            "TableD",
            "TableE",
        ):
            assert {
                placement[(spmd_id, rank)]
                for rank in range(graph.spmd(spmd_id).world_size)
            } == {"node-a", "node-b"}, spmd_id


def test_recipe_rejects_different_compiled_reader_and_input_lane_counts():
    graph, refs = _compiled_worker_graph(2, 4, 2, ((0, 1, 2, 3),))
    with pytest.raises(
        ValueError, match="Reader logical lane count must equal Input worker count"
    ):
        _worker_placement_recipe(graph, refs, "balanced")


def test_feasible_policy_balances_each_embedding_table_across_two_nodes():
    dp = 16
    readers = 32
    groups = (tuple(range(0, readers, 2)), tuple(range(1, readers, 2)))
    graph, refs = _compiled_worker_graph(dp, readers, dp, groups)
    node_resources = {
        "CPU": Decimal(166),
        "GPU": Decimal(8),
        "memory": Decimal(103725028066),
    }
    cluster = ClusterSnapshot.from_nodes(
        ClusterNode(node_id=node, resources=node_resources)
        for node in ("node-a", "node-b")
    )
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        inventory=PlacementInventory.from_compiled_graph(graph),
        recipe=_worker_placement_recipe(graph, refs, "feasible"),
        snapshot=cluster,
    )
    counts = {
        spmd_id: {node: 0 for node in ("node-a", "node-b")}
        for spmd_id in (
            "Input",
            "Compute",
            "TableA",
            "TableB",
            "TableC",
            "TableD",
            "TableE",
        )
    }
    for assignment in plan.assignments:
        if assignment.spmd_id in counts:
            counts[assignment.spmd_id][assignment.node_id] += 1

    assert counts == {
        spmd_id: {"node-a": dp // 2, "node-b": dp // 2} for spmd_id in counts
    }
