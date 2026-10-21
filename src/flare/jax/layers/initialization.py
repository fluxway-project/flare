from __future__ import annotations

from flax import nnx


def require_rngs(rngs: nnx.Rngs | None) -> nnx.Rngs:
    if rngs is None:
        raise ValueError("parameterized layer construction requires rngs")
    return rngs


__all__ = ["require_rngs"]
