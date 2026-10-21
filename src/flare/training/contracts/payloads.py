from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PayloadField:
    name: str
    dtype: str
    shape: tuple[int | None, ...]
    optional: bool = False
    is_dynamic: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"PayloadField requires non-empty name, got {self.name!r}")
        if not self.dtype or not isinstance(self.dtype, str):
            raise ValueError(
                f"PayloadField requires non-empty dtype, got {self.dtype!r}"
            )
        if not isinstance(self.shape, tuple):
            raise TypeError(
                f"PayloadField shape must be a tuple, got {type(self.shape).__name__}"
            )


@dataclass(frozen=True)
class PayloadSchema:
    name: str
    fields: tuple[PayloadField, ...]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(
                f"PayloadSchema requires non-empty name, got {self.name!r}"
            )
        if not isinstance(self.fields, tuple):
            raise TypeError(
                f"PayloadSchema fields must be a tuple, got {type(self.fields).__name__}"
            )
        names = set()
        for f in self.fields:
            if not isinstance(f, PayloadField):
                raise TypeError(f"Field must be PayloadField, got {type(f).__name__}")
            if f.name in names:
                raise ValueError(
                    f"Duplicate field name {f.name!r} in PayloadSchema {self.name!r}"
                )
            names.add(f.name)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


__all__ = ["PayloadField", "PayloadSchema"]
