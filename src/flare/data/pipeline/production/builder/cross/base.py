"""Cross-feature registry — maps config type names to factory callables."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flare.data.pipeline.production.builder.cross.protocol import (
        FeaCrossExpr,
        FidCrossExpr,
    )

# Registry: type_name -> factory callable
_FEA_REGISTRY: dict[str, Callable[..., FeaCrossExpr]] = {}
_FID_REGISTRY: dict[str, Callable[..., FidCrossExpr]] = {}


def register_fea(type_name: str):
    """Decorator to register a FeaCrossExpr factory under ``type_name``."""

    def decorator(cls):
        _FEA_REGISTRY[type_name] = cls
        return cls

    return decorator


def register_fid(type_name: str):
    """Decorator to register a FidCrossExpr factory under ``type_name``."""

    def decorator(cls):
        _FID_REGISTRY[type_name] = cls
        return cls

    return decorator


def get_fea_factory(type_name: str) -> Callable[..., FeaCrossExpr]:
    """Look up a registered FeaCrossExpr factory by type name."""
    if type_name not in _FEA_REGISTRY:
        raise KeyError(
            f"Unknown fea cross type: {type_name!r}. Available: {list(_FEA_REGISTRY)}"
        )
    return _FEA_REGISTRY[type_name]


def get_fid_factory(type_name: str) -> Callable[..., FidCrossExpr]:
    """Look up a registered FidCrossExpr factory by type name."""
    if type_name not in _FID_REGISTRY:
        raise KeyError(
            f"Unknown fid cross type: {type_name!r}. Available: {list(_FID_REGISTRY)}"
        )
    return _FID_REGISTRY[type_name]


__all__ = [
    "get_fea_factory",
    "get_fid_factory",
    "register_fea",
    "register_fid",
]
