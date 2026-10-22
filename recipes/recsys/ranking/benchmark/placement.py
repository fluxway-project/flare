from __future__ import annotations

from fluxway import (
    PlacementRecipe,
    SPMDRef,
    lexicographic,
    max_min,
    occupancy,
    subseteq,
    support,
    workers,
)


def build_data_throughput_placement(
    *,
    recipe_id: str,
    reader_spmd: SPMDRef,
    input_spmd: SPMDRef,
    sink_spmd: SPMDRef,
) -> PlacementRecipe:
    reader_workers = workers(reader_spmd)
    return PlacementRecipe(
        recipe_id=recipe_id,
        constraints=(
            subseteq(support(reader_workers), support(input_spmd)),
            subseteq(support(input_spmd), support(sink_spmd)),
            subseteq(support(sink_spmd), support(input_spmd)),
        ),
        objective=lexicographic(
            max_min(occupancy(reader_workers)),
            max_min(occupancy(input_spmd)),
            max_min(occupancy(sink_spmd)),
        ),
    )


__all__ = ["build_data_throughput_placement"]
