import json
from pathlib import Path


def _read_vocabulary(path: Path) -> dict[str, int]:
    values = [
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not values:
        raise ValueError(f"category vocabulary is empty: {path}")
    mapping = {value: index for index, value in enumerate(values)}
    if len(mapping) != len(values):
        raise ValueError(f"category vocabulary contains duplicate values: {path}")
    return mapping


def load_model_category_values(
    model_conf_dir: Path, category_dir: Path
) -> tuple[dict[str, dict[str, int]], frozenset[str]]:
    config = json.loads(
        (model_conf_dir / "model_config.json").read_text(encoding="utf-8")
    )
    declared = config["category_file"]
    if not isinstance(declared, dict):
        raise TypeError("category_file must be an object")
    vocabularies: dict[str, dict[str, int]] = {}
    values: dict[str, dict[str, int]] = {}
    auto: set[str] = set()
    for feature, filename in declared.items():
        if filename == "auto":
            auto.add(feature)
            continue
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"invalid category file for {feature!r}")
        if filename not in vocabularies:
            vocabularies[filename] = _read_vocabulary(category_dir / filename)
        values[feature] = vocabularies[filename]
    return values, frozenset(auto)
