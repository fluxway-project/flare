from __future__ import annotations

from collections.abc import Container, Mapping
from typing import Literal

from fluxway import (
    PlacementRecipe,
    SPMDRef,
    affinity_constraints,
    equal,
    incident_spmds,
    isolated_spmds,
    lexicographic,
    max_min,
    occupancy,
    rank,
    ranks,
    support,
    weighted_dominant_share,
    workers,
)
from fluxway.common.compiled_graph import CompiledGraph


def _affinity_sparse_tables(
    sparse_table_spmds: Mapping[str, SPMDRef],
    excluded_tables: Container[str] = frozenset(),
) -> tuple[SPMDRef, ...]:
    names = tuple(
        sorted(name for name in sparse_table_spmds if name not in excluded_tables)
    )
    if not names:
        raise ValueError(
            "affinity placement requires at least one regular sparse table"
        )
    return tuple(sparse_table_spmds[name] for name in names)


def build_affinity_placement(
    *,
    recipe_id: str,
    compiled_graph: CompiledGraph,
    reader_spmd: SPMDRef,
    input_spmd: SPMDRef,
    sparse_table_spmds: Mapping[str, SPMDRef],
    compute_spmd: SPMDRef,
    policy: Literal["balanced", "feasible"] = "balanced",
    excluded_affinity_tables: Container[str] = frozenset(),
) -> PlacementRecipe:
    if not isinstance(compiled_graph, CompiledGraph):
        raise TypeError("compiled_graph must be a Fluxway CompiledGraph")
    reader_layout = (
        compiled_graph.spmd(reader_spmd.spmd_id).operator("read").execution_layout
    )
    input_world_size = compiled_graph.spmd(input_spmd.spmd_id).world_size
    if reader_layout.logical_mesh.size != input_world_size:
        raise ValueError(
            "Reader logical lane count must equal Input worker count: "
            f"{reader_layout.logical_mesh.size} != {input_world_size}"
        )
    reader_lane_constraints = tuple(
        equal(
            support(
                ranks(
                    reader_spmd,
                    tuple(group.physical_rank(lane) for group in reader_layout.groups),
                )
            ),
            support(rank(input_spmd, lane)),
        )
        for lane in range(input_world_size)
    )
    affinity_tables = _affinity_sparse_tables(
        sparse_table_spmds, excluded_affinity_tables
    )
    all_sparse_tables = tuple(
        sparse_table_spmds[name] for name in sorted(sparse_table_spmds)
    )
    reader_workers = workers(reader_spmd)
    affinity_relations = affinity_constraints()
    if policy == "balanced":
        objective = lexicographic(
            max_min(weighted_dominant_share(incident_spmds(affinity_relations))),
            max_min(weighted_dominant_share(isolated_spmds(affinity_relations))),
            max_min(occupancy(reader_workers)),
            max_min(occupancy(input_spmd)),
        )
    elif policy == "feasible":
        # ``feasible`` used to optimize only Reader occupancy.  The support
        # constraint below merely required all SPMDs to touch the same set of
        # nodes, so the solver was free to put every embedding rank on one
        # node (and routinely did so).  Keep this policy cheaper than the
        # weighted affinity objective, but make its placement useful for
        # distributed training by maximizing the per-node minimum occupancy
        # for every participating worker class.
        objective = lexicographic(
            max_min(occupancy(reader_workers)),
            max_min(occupancy(input_spmd)),
            *(max_min(occupancy(table)) for table in all_sparse_tables),
            max_min(occupancy(compute_spmd)),
        )
    else:
        raise ValueError(f"unknown placement policy: {policy!r}")
    return PlacementRecipe(
        recipe_id=recipe_id,
        constraints=(
            equal(
                support(reader_workers),
                support(input_spmd),
                support(affinity_tables),
                support(compute_spmd),
            ),
            *reader_lane_constraints,
        ),
        objective=objective,
    )


__all__ = ["build_affinity_placement"]
