from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).parents[2]
_RECSYS = _ROOT / "recipes" / "recsys" / "ranking"
_CONFIG_PATHS = _ROOT / "recipes" / "config" / "paths.py"


def test_recsys_shared_owner_exists() -> None:
    for relative_path in (
        "schema/feature_metadata.py",
        "schema/labels.py",
        "schema/storage.py",
        "graph/forward.py",
    ):
        assert (_RECSYS / relative_path).is_file()
    assert not (_ROOT / "recipes" / "_shared").exists()


def test_recipe_specific_duplicate_modules_are_absent() -> None:
    for relative_path in (
        "data/feature_metadata.py",
        "data/labels.py",
        "data/storage_contract.py",
        "training/forward.py",
        "training/losses.py",
        "tasks/ranking/forward.py",
        "util/training_monitor.py",
        "conf/conf_common/category_feature/parse.py",
    ):
        assert not (_ROOT / "recipes" / "ae11" / relative_path).exists()
        assert not (_ROOT / "recipes" / "whw" / relative_path).exists()
    assert not (
        _ROOT / "examples" / "recsys_ranking" / "training" / "forward.py"
    ).exists()
    assert not (
        _ROOT / "examples" / "recsys_ranking" / "training" / "losses.py"
    ).exists()


def test_recsys_loss_and_forward_have_one_owner() -> None:
    assert (_RECSYS / "losses.py").is_file()
    assert (_RECSYS / "graph" / "forward.py").is_file()
    for root in (_ROOT / "recipes", _ROOT / "examples"):
        matches = tuple(
            path
            for path in root.rglob("losses.py")
            if path.name == "losses.py" and "ranking" not in path.parts
        )
        assert not matches, matches


def test_generic_pipeline_tests_have_one_owner() -> None:
    for relative_path in (
        "test_feature_layout.py",
        "test_feature_output_schema.py",
        "test_gravitino_pyiceberg_s3.py",
    ):
        matches = tuple(
            path
            for path in _ROOT.rglob(relative_path)
            if not {".git", "build", "dist", "__pycache__"}.intersection(path.parts)
        )
        assert len(matches) == 1, matches


def test_recipe_config_paths_have_one_owner() -> None:
    assert _CONFIG_PATHS.is_file()
    for relative_path in (
        "recipes/ae11/conf/conf_common/base_params.py",
        "recipes/whw/conf/conf_common/base_params.py",
        "examples/recsys_ranking/conf/conf_common/base_params.py",
    ):
        assert not (_ROOT / relative_path).exists()
    for root in (_ROOT / "recipes", _ROOT / "examples"):
        assert not any(
            "base_params" in path.name
            for path in root.rglob("*.py")
            if ".git" not in path.parts
        )
