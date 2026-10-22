from __future__ import annotations

import ast
from pathlib import Path

from recipes.recsys.ranking.graph.preset import RecsysBackboneBindings

ROOT = Path(__file__).parents[2]


def _calls_named(source: str, name: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_input_graph_calls_forward_owner_key_dtype() -> None:
    paths = (
        ROOT / "recipes/recsys/ranking/graph/pipeline.py",
        ROOT / "recipes/recsys/ranking/benchmark/pipeline.py",
    )
    for path in paths:
        calls = _calls_named(path.read_text(encoding="utf-8"), "input_spmd")
        assert calls, path
        for call in calls:
            keywords = {keyword.arg for keyword in call.keywords}
            assert "key_dtype" in keywords, path


def test_ranking_preset_input_static_args_include_owner_key_dtype() -> None:
    bindings = RecsysBackboneBindings(
        training_spmd=object,
        evaluation_spmd=object,
        sparse_groups=("feature_dim_8",),
        side_info_groups=("side_info",),
    )
    static_args = bindings.input_static_args(
        input_node_output=object,
        key_dtype="uint64",
    )
    assert static_args["key_dtype"] == "uint64"


def test_ranking_preset_evaluation_static_args_include_required_worker_inputs() -> None:
    bindings = RecsysBackboneBindings(
        training_spmd=object,
        evaluation_spmd=object,
        sparse_groups=("feature_dim_8",),
        side_info_groups=("side_info",),
    )
    weight_source = object()
    static_args = bindings.evaluation_static_args(
        model_definition=object(),
        ranking_output=object,
        sparse_group_widths={"feature_dim": (8, 1)},
        sparse_tensor_backend="numpy",
        forward_fn=object(),
        weight_source=weight_source,
        sparse_embedding_dtype="float32",
    )
    assert static_args["weight_source"] is weight_source
    assert static_args["sparse_embedding_dtype"] == "float32"
