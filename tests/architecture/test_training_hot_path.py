from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import flare.runtime.sparse_precision as sparse_precision
from flare.runtime.sparse_precision import prepare_sparse_native_gradient


ROOT = Path(__file__).resolve().parents[2]


def _method_nodes(path: str, names: set[str]) -> list[ast.AST]:
    tree = ast.parse((ROOT / path).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]


def test_sparse_embedding_trace_and_readback_are_opt_in() -> None:
    methods = _method_nodes(
        "src/flare/runtime/fluxway/workers/dynamic_embedding.py",
        {"lookup", "_accumulate_gradients", "update", "flush_pending_updates"},
    )
    assert {method.name for method in methods} == {
        "lookup",
        "_accumulate_gradients",
        "update",
        "flush_pending_updates",
    }
    source = (ROOT / "src/flare/runtime/fluxway/workers/dynamic_embedding.py").read_text()
    assert 'getattr(self, "execution_tracer", None)' in source
    assert 'if validate_tensor_values:' in source
    assert "_read_embeddings_for_audit" in source
    assert "_traced_gradients" in source


def test_default_sparse_gradient_path_does_not_validate_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("finite validation must be opt-in")

    monkeypatch.setattr(sparse_precision, "require_finite_sparse_values", fail_if_called)
    result = prepare_sparse_native_gradient(
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        transport_dtype="float32",
        native_dtype="float32",
    )
    np.testing.assert_array_equal(result, np.asarray([[1.0, 2.0]], dtype=np.float32))


def test_explicit_sparse_value_validation_is_available() -> None:
    with pytest.raises(FloatingPointError, match="sparse_native_gradient"):
        prepare_sparse_native_gradient(
            np.asarray([[np.nan, np.inf]], dtype=np.float32),
            transport_dtype="float32",
            native_dtype="float32",
            validate_values=True,
        )
