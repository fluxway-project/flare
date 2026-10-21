from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Literal

from fluxway import ResourceProfile, ResourceScale

GiB = 1024**3


DEFAULT_ROLE_PROFILES: Mapping[str, ResourceProfile] = {
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
        base={"CPU": Decimal("1.01"), "memory": Decimal("2.1") * GiB},
        ray_component_limits={"CPU": Decimal(2), "memory": Decimal(2) * GiB},
    ),
}

CPU_COMPUTE_PROFILE = ResourceProfile(base={"CPU": 2, "memory": 2 * GiB})


class ResourceProfiles:
    def __init__(self, roles: Mapping[str, ResourceProfile]) -> None:
        self._roles = dict(roles)

    def get_role(self, role_name: str) -> ResourceProfile:
        try:
            return self._roles[role_name]
        except KeyError as error:
            raise KeyError(f"Role {role_name!r} has no ResourceProfile") from error


def resolve_resource_profiles(
    worker_counts: Mapping[str, int],
    *,
    compute_device: Literal["cpu", "gpu"] = "gpu",
) -> ResourceProfiles:
    if compute_device not in {"cpu", "gpu"}:
        raise ValueError(
            f"compute_device must be 'cpu' or 'gpu', got {compute_device!r}"
        )
    invalid = {
        role: count
        for role, count in worker_counts.items()
        if not isinstance(count, int) or count <= 0
    }
    if invalid:
        raise ValueError(f"worker counts must be positive integers: {invalid}")
    missing = sorted(set(worker_counts) - set(DEFAULT_ROLE_PROFILES) - {"driver"})
    if missing:
        raise KeyError(f"worker roles have no ResourceProfile: {missing}")
    roles = {
        role: (
            CPU_COMPUTE_PROFILE
            if role == "compute" and compute_device == "cpu"
            else DEFAULT_ROLE_PROFILES[role]
        )
        for role in worker_counts
        if role in DEFAULT_ROLE_PROFILES
    }
    return ResourceProfiles(roles)


__all__ = [
    "CPU_COMPUTE_PROFILE",
    "DEFAULT_ROLE_PROFILES",
    "ResourceProfiles",
    "resolve_resource_profiles",
]
