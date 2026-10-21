from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx

from flare.jax.roles import ConstantBuffer, Fp32Param

_PARAM_DTYPES = frozenset(
    {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)}
)


def _path_string(path: tuple[Any, ...]) -> str:
    return jtu.keystr(path)


def _cast_array(value: Any, dtype: jnp.dtype) -> Any:
    if isinstance(value, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(
            value.shape, dtype, sharding=getattr(value, "sharding", None)
        )
    return jnp.asarray(value).astype(dtype)


def _canonical_graph_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, _canonical_graph_value(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    (repr(key), _canonical_graph_value(item))
                    for key, item in value.items()
                ),
                key=lambda entry: entry[0],
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_graph_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_canonical_graph_value(item) for item in value))
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if slots:
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (name, _canonical_graph_value(getattr(value, name))) for name in slots
            ),
        )
    if hasattr(value, "__dict__"):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _canonical_graph_value(vars(value)),
        )
    return repr(value)


class NnxModelDefinition:
    """Canonical NNX graph and model-state schema shared by train and eval."""

    def __init__(
        self,
        module_initializer: Callable[[], nnx.Module],
        *,
        sparse_groups: Iterable[str] = (),
    ) -> None:
        if not callable(module_initializer):
            raise TypeError("module_initializer must be callable")
        normalized_sparse_groups = tuple(sparse_groups)
        if any(
            not isinstance(group, str) or not group
            for group in normalized_sparse_groups
        ):
            raise ValueError("sparse_groups must contain non-empty strings")
        if len(set(normalized_sparse_groups)) != len(normalized_sparse_groups):
            raise ValueError("sparse_groups must be unique")
        abstract_module = nnx.eval_shape(module_initializer)
        if not isinstance(abstract_module, nnx.Module):
            raise TypeError("module_initializer must return flax.nnx.Module")
        graphdef, param_state, non_param_state = nnx.split(
            abstract_module, nnx.Param, ...
        )
        normalized_params = jtu.tree_map(
            lambda leaf: self._normalize_param_leaf((), leaf), param_state
        )
        normalized_module = nnx.merge(
            cast(Any, graphdef), normalized_params, non_param_state
        )
        if not isinstance(normalized_module, nnx.Module):
            raise TypeError("model graph did not materialize an NNX module")
        _, model_state = nnx.split(normalized_module)
        self._module_initializer = module_initializer
        self._module_graphdef = graphdef
        self._module_graph_signature = _canonical_graph_value(graphdef)
        self._module_state = model_state
        self._param_paths = frozenset(
            _path_string(path) for path, _ in jtu.tree_flatten_with_path(param_state)[0]
        )
        self._sparse_groups = frozenset(normalized_sparse_groups)

    @classmethod
    def from_initializer(
        cls,
        module_initializer: Callable[[], nnx.Module],
        *,
        sparse_groups: Iterable[str] = (),
    ) -> NnxModelDefinition:
        return cls(module_initializer, sparse_groups=sparse_groups)

    @classmethod
    def from_factory(
        cls,
        factory: Any,
        *,
        sparse_groups: Iterable[str] = (),
    ) -> NnxModelDefinition:
        initializer = getattr(factory, "create_module", None)
        if not callable(initializer):
            raise TypeError("factory must expose callable create_module")
        return cls(
            cast(Callable[[], nnx.Module], initializer),
            sparse_groups=sparse_groups,
        )

    @property
    def graphdef(self) -> Any:
        return self._module_graphdef

    def abstract_model_state(self) -> Any:
        return self._module_state

    @property
    def param_paths(self) -> frozenset[str]:
        return self._param_paths

    @property
    def sparse_groups(self) -> frozenset[str]:
        return self._sparse_groups

    def initialize(self, *args: Any, **kwargs: Any) -> nnx.Module:
        module = self._module_initializer(*args, **kwargs)
        if not isinstance(module, nnx.Module):
            raise TypeError("module initializer must return flax.nnx.Module")
        graphdef, state = nnx.split(module)
        self._validate_graph(graphdef)
        normalized_state = self._normalize_state(state, _PARAM_DTYPES)
        materialized = nnx.merge(cast(Any, self._module_graphdef), normalized_state)
        if not isinstance(materialized, nnx.Module):
            raise TypeError("model graph did not materialize an NNX module")
        return materialized

    def materialize(
        self,
        model_state: Any,
        *,
        allowed_param_dtypes: Iterable[jnp.dtype] = _PARAM_DTYPES,
    ) -> nnx.Module:
        normalized = self._normalize_state(model_state, allowed_param_dtypes)
        module = nnx.merge(cast(Any, self._module_graphdef), normalized)
        if not isinstance(module, nnx.Module):
            raise TypeError("model graph did not materialize an NNX module")
        return module

    def validate_module(self, module: nnx.Module) -> None:
        if not isinstance(module, nnx.Module):
            raise TypeError("module must be flax.nnx.Module")
        graphdef, state = nnx.split(module)
        self._validate_graph(graphdef)
        self._validate_state(state, _PARAM_DTYPES)

    def validate_state(
        self,
        model_state: Any,
        *,
        allowed_param_dtypes: Iterable[jnp.dtype] = _PARAM_DTYPES,
    ) -> None:
        self._validate_state(model_state, allowed_param_dtypes)

    def _validate_graph(self, graphdef: Any) -> None:
        if _canonical_graph_value(graphdef) != self._module_graph_signature:
            raise ValueError("NNX module graph differs from the model definition")

    def _normalize_state(
        self, model_state: Any, allowed_param_dtypes: Iterable[jnp.dtype]
    ) -> Any:
        self._validate_state(model_state, allowed_param_dtypes)
        return jtu.tree_map_with_path(
            lambda path, leaf: (
                _cast_array(leaf, jnp.float32)
                if _path_string(path) in self._param_paths
                else leaf
            ),
            model_state,
        )

    def _validate_state(
        self, model_state: Any, allowed_param_dtypes: Iterable[jnp.dtype]
    ) -> None:
        expected_entries = jtu.tree_flatten_with_path(self._module_state)[0]
        actual_entries = jtu.tree_flatten_with_path(model_state)[0]
        if jtu.tree_structure(self._module_state) != jtu.tree_structure(model_state):
            raise ValueError("NNX model state tree differs from model definition")
        allowed = frozenset(jnp.dtype(dtype) for dtype in allowed_param_dtypes)
        for (expected_path, expected_leaf), (actual_path, actual_leaf) in zip(
            expected_entries, actual_entries, strict=True
        ):
            expected_name = _path_string(expected_path)
            if expected_name != _path_string(actual_path):
                raise ValueError("NNX model state tree differs from model definition")
            if not hasattr(actual_leaf, "shape") or not hasattr(actual_leaf, "dtype"):
                raise TypeError(
                    f"NNX model state leaf {expected_name!r} lacks shape/dtype"
                )
            if tuple(actual_leaf.shape) != tuple(expected_leaf.shape):
                raise ValueError(
                    f"NNX model state leaf {expected_name!r} shape differs"
                )
            actual_dtype = jnp.dtype(actual_leaf.dtype)
            expected_dtype = jnp.dtype(expected_leaf.dtype)
            if expected_name in self._param_paths:
                if actual_dtype not in allowed:
                    raise TypeError(
                        f"NNX Param leaf {expected_name!r} must be float16, bfloat16, or float32, got {actual_dtype}"
                    )
            elif actual_dtype != expected_dtype:
                raise ValueError(
                    f"NNX model state leaf {expected_name!r} dtype differs"
                )

    @staticmethod
    def _normalize_param_leaf(path: tuple[Any, ...], leaf: Any) -> Any:
        if not hasattr(leaf, "dtype"):
            raise TypeError(f"NNX Param leaf {_path_string(path)!r} lacks dtype")
        dtype = jnp.dtype(leaf.dtype)
        if dtype not in _PARAM_DTYPES:
            raise TypeError(
                f"NNX Param leaf {_path_string(path)!r} must be float16, bfloat16, or float32, got {dtype}"
            )
        return _cast_array(leaf, jnp.float32)


__all__ = ["ConstantBuffer", "Fp32Param", "NnxModelDefinition"]
