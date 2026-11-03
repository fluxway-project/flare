from __future__ import annotations

from decimal import Decimal

import pytest

from recipes.recsys.ranking.resources import resolve_ranking_resource_profiles


def test_metric_ray_parallelism_is_a_ranking_override() -> None:
    default = resolve_ranking_resource_profiles({"metrics": 1}).get_role("metrics")
    scaled = resolve_ranking_resource_profiles(
        {"metrics": 1}, metric_ray_parallelism=4
    ).get_role("metrics")
    assert default.ray_component_limits["CPU"] == Decimal(2)
    assert scaled.ray_component_limits["CPU"] == Decimal(4)
    assert scaled.base["CPU"] == default.base["CPU"]
    with pytest.raises(ValueError, match="metric_ray_parallelism"):
        resolve_ranking_resource_profiles({"metrics": 1}, metric_ray_parallelism=0)


def test_input_cpu_request_is_a_ranking_override() -> None:
    default = resolve_ranking_resource_profiles({"input": 1}).get_role("input")
    scaled = resolve_ranking_resource_profiles({"input": 1}, input_cpus=4).get_role(
        "input"
    )
    assert default.base["CPU"] == Decimal(2)
    assert scaled.base["CPU"] == Decimal(4)
    assert scaled.base["memory"] == default.base["memory"]
    with pytest.raises(ValueError, match="input_cpus"):
        resolve_ranking_resource_profiles({"input": 1}, input_cpus=0)
