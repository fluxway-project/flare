from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PinnedSnapshot:
    snapshot_id: int

    def __post_init__(self) -> None:
        if type(self.snapshot_id) is not int or self.snapshot_id <= 0:
            raise ValueError("PinnedSnapshot snapshot_id must be positive")


@dataclass(frozen=True, slots=True)
class ContinuousAppend:
    branch: str = "main"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.branch, str)
            or not self.branch
            or self.branch.strip() != self.branch
        ):
            raise ValueError("ContinuousAppend branch must be canonical")
