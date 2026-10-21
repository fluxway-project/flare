import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from flare.config.discovery import find_named_directory
from flare.config.loader import LayeredConfigLoader, dict_to_cli_args


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class DummyConfig:
    model_name: str
    batch_size: int = 1024
    enable_dropout: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def test_dict_to_cli_args_conversion():
    d = {
        "model_name": "test_net",
        "batch_size": 2048,
        "enable_dropout": False,
        "optimizer": {
            "lr": 0.01,
            "weight_decay": 0.001,
        },
    }
    args = dict_to_cli_args(d)
    assert "--model-name" in args
    assert "test_net" in args
    assert "--batch-size" in args
    assert "2048" in args
    assert "--no-enable-dropout" in args
    assert "--optimizer.lr" in args
    assert "0.01" in args


def test_layered_config_loader_cli_override(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "model_name": "base_model",
                "batch_size": 2048,
                "optimizer": {"lr": 0.05},
            }
        )
    )

    loader = LayeredConfigLoader()
    result = loader.load(
        DummyConfig,
        args=["--config", str(config_file), "--batch-size", "4096"],
    )
    assert result.model_name == "base_model"
    assert result.batch_size == 4096
    assert result.optimizer.lr == 0.05


def test_layered_config_loader_file_not_found():
    loader = LayeredConfigLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(DummyConfig, args=["--config", "/non/existent/path.json"])


def test_layered_config_loader_invalid_json_type(tmp_path: Path):
    config_file = tmp_path / "invalid.json"
    config_file.write_text("[1, 2, 3]")

    loader = LayeredConfigLoader()
    with pytest.raises(TypeError):
        loader.load(DummyConfig, args=["--config", str(config_file)])


def test_find_named_directory(tmp_path: Path):
    target = tmp_path / "nested" / "my_dir"
    target.mkdir(parents=True)

    found = find_named_directory(tmp_path, "my_dir")
    assert found == target

    with pytest.raises(FileNotFoundError):
        find_named_directory(tmp_path, "unknown_dir")
