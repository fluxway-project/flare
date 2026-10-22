from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from fluxway import PlacementRecipe
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


def test_fluxway_native_placement_recipe() -> None:
    graph, refs = _compiled_ranking_graph(2, 4, 2, ((0, 2), (1, 3)))
    recipe = build_affinity_placement(
        recipe_id="native",
        compiled_graph=graph,
        reader_spmd=refs["Reader"],
        input_spmd=refs["Input"],
        sparse_table_spmds={"item": refs["Job"], "user": refs["Geek"]},
        compute_spmd=refs["Ranking"],
    )
    assert isinstance(recipe, PlacementRecipe)
    assert recipe.recipe_id == "native"
    assert recipe.constraints
    assert recipe.objective is not None


def test_fluxway_native_placement_excludes_requested_table() -> None:
    graph, refs = _compiled_ranking_graph(2, 4, 2, ((0, 2), (1, 3)))
    recipe = build_affinity_placement(
        recipe_id="native",
        compiled_graph=graph,
        reader_spmd=refs["Reader"],
        input_spmd=refs["Input"],
        sparse_table_spmds={"item": refs["Job"], "user": refs["Geek"]},
        compute_spmd=refs["Ranking"],
        excluded_affinity_tables={"user"},
    )
    assert recipe.constraints


def _compiled_ranking_graph(dp, readers, sparse, groups):
    profiles = resolve_resource_profiles(
        {
            "reader": readers,
            "input": dp,
            "compute": dp,
            "table": 5 * sparse,
            "metrics": 1,
        }
    )
    role_counts = {
        "Reader": ("reader", readers),
        "Input": ("input", dp),
        "Ranking": ("compute", dp),
        "Metrics": ("metrics", 1),
        **{
            name: ("table", sparse) for name in ("Exp", "Hashed", "Job", "Geek", "Boss")
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


def _ranking_recipe(graph, refs, policy):
    return build_affinity_placement(
        recipe_id="reader-lanes",
        compiled_graph=graph,
        reader_spmd=refs["Reader"],
        input_spmd=refs["Input"],
        sparse_table_spmds={
            "exp_id": refs["Exp"],
            "hashed_sparse": refs["Hashed"],
            "job_id": refs["Job"],
            "isolated_table_a": refs["Geek"],
            "isolated_table_b": refs["Boss"],
        },
        compute_spmd=refs["Ranking"],
        excluded_affinity_tables=frozenset({"isolated_table_a", "isolated_table_b"}),
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
    graph, refs = _compiled_ranking_graph(dp, readers, sparse, groups)
    total = {"CPU": Decimal(48), "GPU": Decimal(4), "memory": Decimal(103725028066)}
    cluster = ClusterSnapshot.from_nodes(
        ClusterNode(
            node_id=node,
            resources={key: value * share for key, value in total.items()},
        )
        for node, share in (("node-a", fraction), ("node-b", 1 - fraction))
    )
    recipe = _ranking_recipe(graph, refs, policy)
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


def test_recipe_rejects_different_compiled_reader_and_input_lane_counts():
    graph, refs = _compiled_ranking_graph(2, 4, 2, ((0, 1, 2, 3),))
    with pytest.raises(
        ValueError, match="Reader logical lane count must equal Input worker count"
    ):
        _ranking_recipe(graph, refs, "balanced")
