"""Provider-declared data identities, independent of storage and governance."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssetRef:
    namespace: str
    name: str
    version: str | None = None

    def __post_init__(self) -> None:
        for field in ("namespace", "name", "version"):
            value = getattr(self, field)
            if field == "version" and value is None:
                continue
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"asset {field} must be a canonical non-empty string")
