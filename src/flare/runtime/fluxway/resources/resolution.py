from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from fluxway import ResourceProfile


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
    role_profiles: Mapping[str, ResourceProfile],
    *,
    compute_device: Literal["cpu", "gpu"] = "gpu",
    device_overrides: Mapping[str, Mapping[str, ResourceProfile]] | None = None,
) -> ResourceProfiles:
    """Resolve application-provided role profiles for a worker layout.

    Role names and their resource reservations belong to the application or
    recipe layer.  The runtime layer only validates the layout and selects an
    optional device-specific override, so it does not need to know about
    reader, input, metrics, or any other application role.
    """
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
    missing = sorted(set(worker_counts) - set(role_profiles) - {"driver"})
    if missing:
        raise KeyError(f"worker roles have no ResourceProfile: {missing}")
    overrides = (device_overrides or {}).get(compute_device, {})
    roles = {
        role: overrides.get(role, role_profiles[role])
        for role in worker_counts
        if role in role_profiles
    }
    return ResourceProfiles(roles)


__all__ = [
    "ResourceProfiles",
    "resolve_resource_profiles",
]
