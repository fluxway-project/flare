"""Resource profiles for the ranking recipe and its metric pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Literal

from fluxway import ResourceProfile, ResourceScale

from flare.runtime.fluxway.resources.resolution import (
    ResourceProfiles,
    resolve_resource_profiles,
)

GiB = 1024**3


DEFAULT_RANKING_ROLE_PROFILES: Mapping[str, ResourceProfile] = {
    "reader": ResourceProfile(base={"CPU": 1, "memory": 4 * GiB}),
    "input": ResourceProfile(base={"CPU": 2, "memory": 8 * GiB}),
    "compute": ResourceProfile(base={"CPU": 2, "GPU": 1, "memory": 2 * GiB}),
    "table": ResourceProfile(
        base={"CPU": 1, "memory": GiB},
        scale=ResourceScale(
            step={"CPU": 1, "memory": 6 * GiB},
            max_steps=63,
            weight=2,
        ),
    ),
    "metrics": ResourceProfile(
        base={"CPU": Decimal("1.01"), "memory": Decimal("2.1") * GiB}
    ),
}

CPU_RANKING_COMPUTE_PROFILE = ResourceProfile(base={"CPU": 2, "memory": 2 * GiB})


def resolve_ranking_resource_profiles(
    worker_counts: Mapping[str, int],
    *,
    compute_device: Literal["cpu", "gpu"] = "gpu",
    input_cpus: int | None = None,
    metric_ray_parallelism: int = 2,
) -> ResourceProfiles:
    """Resolve generic roles, then apply ranking-only worker reservations."""
    if input_cpus is not None and (type(input_cpus) is not int or input_cpus <= 0):
        raise ValueError("input_cpus must be a positive integer")
    if type(metric_ray_parallelism) is not int or metric_ray_parallelism <= 0:
        raise ValueError("metric_ray_parallelism must be a positive integer")

    profiles = resolve_resource_profiles(
        worker_counts,
        DEFAULT_RANKING_ROLE_PROFILES,
        compute_device=compute_device,
        device_overrides={"cpu": {"compute": CPU_RANKING_COMPUTE_PROFILE}},
    )
    roles: dict[str, ResourceProfile] = {
        role: profiles.get_role(role) for role in worker_counts if role != "driver"
    }
    if input_cpus is not None and "input" in roles:
        profile = roles["input"]
        roles["input"] = profile.model_copy(
            update={"base": {**profile.base, "CPU": Decimal(input_cpus)}}
        )
    if "metrics" in roles:
        profile = roles["metrics"]
        roles["metrics"] = profile.model_copy(
            update={
                "ray_component_limits": {
                    **profile.ray_component_limits,
                    "CPU": Decimal(metric_ray_parallelism),
                }
            }
        )
    return ResourceProfiles(roles)


__all__ = [
    "CPU_RANKING_COMPUTE_PROFILE",
    "DEFAULT_RANKING_ROLE_PROFILES",
    "resolve_ranking_resource_profiles",
]
