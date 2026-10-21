from __future__ import annotations

import inspect

import pytest
import tyro

from flare.jax.training.precision import PrecisionPolicy
from recipes.ae11.models.registry import load_model_module
from recipes.ae11.util.conf_loader import load_conf
from recipes.ae11.util.config import ModelEnum as Ae11ModelEnum
from recipes.ae11.util.config import TrainConfig as Ae11TrainConfig
from recipes.whw.util.config import ModelEnum as WhwModelEnum
from recipes.whw.util.config import TrainConfig as WhwTrainConfig


@pytest.mark.parametrize(
    ("config_type", "model"),
    [
        (Ae11TrainConfig, next(iter(Ae11ModelEnum))),
        (WhwTrainConfig, next(iter(WhwModelEnum))),
    ],
)
def test_train_config_precision_defaults_to_fp32(config_type, model) -> None:
    config = config_type(conf_version=model, storage_format="iceberg")
    assert config.precision is PrecisionPolicy.FP32


@pytest.mark.parametrize(
    ("config_type", "model"),
    [
        (Ae11TrainConfig, next(iter(Ae11ModelEnum))),
        (WhwTrainConfig, next(iter(WhwModelEnum))),
    ],
)
def test_train_config_accepts_bf16(config_type, model) -> None:
    config = config_type(
        conf_version=model, storage_format="iceberg", precision=PrecisionPolicy.BF16
    )
    assert config.precision is PrecisionPolicy.BF16


@pytest.mark.parametrize(
    ("config_type", "model"),
    [
        (Ae11TrainConfig, next(iter(Ae11ModelEnum))),
        (WhwTrainConfig, next(iter(WhwModelEnum))),
    ],
)
def test_train_config_accepts_fp16(config_type, model) -> None:
    config = config_type(
        conf_version=model, storage_format="iceberg", precision=PrecisionPolicy.FP16
    )
    assert config.precision is PrecisionPolicy.FP16


@pytest.mark.parametrize(
    ("config_type", "model"),
    [
        (Ae11TrainConfig, next(iter(Ae11ModelEnum))),
        (WhwTrainConfig, next(iter(WhwModelEnum))),
    ],
)
def test_train_config_rejects_precision_strings(config_type, model) -> None:
    with pytest.raises(TypeError, match="precision must be PrecisionPolicy"):
        config_type(conf_version=model, storage_format="iceberg", precision="bf16")


@pytest.mark.parametrize(
    ("config_type", "model"),
    [
        (Ae11TrainConfig, next(iter(Ae11ModelEnum))),
        (WhwTrainConfig, next(iter(WhwModelEnum))),
    ],
)
@pytest.mark.parametrize(
    ("cli_value", "expected"),
    [
        ("fp32", PrecisionPolicy.FP32),
        ("bf16", PrecisionPolicy.BF16),
        ("fp16", PrecisionPolicy.FP16),
    ],
)
def test_train_config_cli_uses_precision_policy_values(
    config_type, model, cli_value, expected
) -> None:
    config = tyro.cli(
        config_type,
        args=[
            "--conf-version",
            model.value,
            "--storage-format",
            "iceberg",
            "--precision",
            cli_value,
        ],
    )
    assert config.precision is expected


def test_reduced_config_uses_canonical_model_and_is_small() -> None:
    production = load_conf("conf_boss_predict_u1_share_s2_v7_ps_refuse_yt_v1_lt_v4")
    test = load_conf("conf_boss_predict_u1_share_s2_v7_ps_refuse_yt_v1_lt_v4_test")
    assert production.conf_path != test.conf_path
    assert test.token_dim == 672
    assert test.ple_num_experts == 4
    assert test.ple_expert_dim == 1024
    assert test.task_mlp_dims == (1024, 512, 1)

    model_module = load_model_module(
        "conf_boss_predict_u1_share_s2_v7_ps_refuse_yt_v1_lt_v4_test"
    )
    assert model_module.__name__.endswith(
        "conf_boss_predict_u1_share_s2_v7_ps_refuse_yt_v1_lt_v4"
    )
    assert inspect.signature(model_module.ModelBuilder.create_optimizer).parameters[
        "learning_rate"
    ].default == pytest.approx(1e-3)
    model_builder = model_module.ModelBuilder(test)
    assert sum(len(features) for features in test.feature_field.values()) < sum(
        len(features) for features in production.feature_field.values()
    )
    assert model_builder.pack_meta.side_info_seq
