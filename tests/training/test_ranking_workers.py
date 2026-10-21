from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest
from fluxway import ResourceProfile

from flare.jax.training.engines.contracts import TrainStepResult
from flare.training.components.steppers import GenericLossAdapter
from flare.training.execution import (
    GradientAccumulationPolicy,
    MacroUpdatePolicy,
    OptimizerUpdateWindow,
    TrainingExecutionPlan,
    ZeroExecutionPolicy,
)
from flare.training.zero import ZeroStage
from recipes.recsys.ranking.graph.bindings import bind_recsys_ranking_training
from recipes.recsys.ranking.graph.losses import RankingLossAdapter
from recipes.recsys.ranking.metrics.eval_metrics import (
    get_metrics_aggregator_spmd_class,
)
from recipes.recsys.ranking.metrics.train_metrics import (
    get_train_metrics_aggregator_spmd_class,
)
from recipes.recsys.ranking.schema.contracts import (
    build_ranking_batch_schema,
)
from sparse_embedding.checkpoint.lance_backend import _LanceCheckpointStorage

from recipes.recsys.ranking.schema.sparse_features import (
    build_sparse_group_feature_widths,
    extract_sparse_base_names,
    extract_sparse_groups_and_dimensions,
    parse_sparse_group_name,
    parse_sparse_workers,
)
from recipes.recsys.ranking.workers.schedule import (
    ranking_phase_schedule,
    ranking_schedule_contract,
)
from recipes.recsys.ranking.workers.train_worker import RecsysBackboneTrainSPMD


def _training_plan() -> TrainingExecutionPlan:
    return TrainingExecutionPlan(
        zero=ZeroExecutionPolicy(ZeroStage.REPLICATED),
        accumulation=GradientAccumulationPolicy(1),
        update=MacroUpdatePolicy(),
    )


def test_training_exports_exist():
    assert callable(RankingLossAdapter)
    assert callable(GenericLossAdapter)
    assert callable(ranking_schedule_contract)
    assert callable(ranking_phase_schedule)
    assert callable(build_ranking_batch_schema)


def test_model_phase_schedule():
    schedule = ranking_phase_schedule("prepare_sparse_materialize")
    assert schedule is not None


def test_sparse_group_widths_require_canonical_pack_meta() -> None:
    categorical_feature_groups = {"feature_dim_4": ("a", "b")}
    feature_lengths = {"a": 2, "b": 3}
    assert build_sparse_group_feature_widths(
        categorical_feature_groups, feature_lengths, ("feature_dim_4",)
    ) == {"feature_dim_4": (2, 3)}

    with pytest.raises(TypeError, match="categorical_feature_groups"):
        build_sparse_group_feature_widths(object(), feature_lengths, ("feature_dim_4",))
    with pytest.raises(KeyError, match="absent"):
        build_sparse_group_feature_widths(
            categorical_feature_groups, feature_lengths, ("missing_dim_4",)
        )


def test_recsys_ranking_train_binding():
    spmd_cls = bind_recsys_ranking_training(
        sparse_groups=("group1_dim_16",),
        side_info_groups=("side_info",),
        runtime_env={},
        resource_profile=ResourceProfile(base={"CPU": 1}),
        dp_size=1,
        pipeline_depth=2,
        execution_plan=_training_plan(),
    )
    assert spmd_cls is not None
    assert spmd_cls.__module__ == RecsysBackboneTrainSPMD.__module__
    assert spmd_cls.__mro__[1] is RecsysBackboneTrainSPMD


def test_public_workers_do_not_export_recsys_benchmark_sink():
    import flare.runtime.fluxway.workers as public_workers

    assert not hasattr(public_workers, "ThroughputSinkSPMD")
    assert not hasattr(public_workers, "get_throughput_sink_spmd_class")


def test_metrics_spmd_binding():
    input_shards = {
        "scores": {"type": "static", "axis": ["dp"]},
        "loss": {"type": "static", "axis": ["dp"]},
        "metric_sampled": {"type": "static", "axis": ["dp"]},
        "labels": {"type": "static", "axis": ["dp"]},
        "group_ids": {"type": "static", "axis": ["dp"]},
        "samples_valid_count": {"type": "static", "axis": []},
    }
    train_metrics_cls = get_train_metrics_aggregator_spmd_class(
        input_sharding=input_shards,
        runtime_env={},
        resource_profile=ResourceProfile(base={"CPU": 1}),
        max_inflight_steps=5,
    )
    assert train_metrics_cls is not None

    eval_metrics_cls = get_metrics_aggregator_spmd_class(
        input_sharding=input_shards,
        runtime_env={},
        resource_profile=ResourceProfile(base={"CPU": 1}),
    )
    assert eval_metrics_cls is not None


def test_metrics_accepts_delivered_optimizer_window_after_update_state_retirement():
    worker = object.__new__(RecsysBackboneTrainSPMD)
    worker.train_eval_metrics_enabled = False
    worker.world_size = 1
    worker._last_update_window = None
    window = OptimizerUpdateWindow(
        micro_step=1,
        accumulation_steps=1,
        at_boundary=True,
        should_apply_update=True,
        optimizer_step=0,
    )

    metrics = worker.materialize_train_metrics(
        optimizer_update_window=np.asarray(window.encode(), dtype=np.int64)
    )

    assert set(metrics) == {"scores", "loss", "metric_sampled"}


def test_metrics_requires_delivered_optimizer_update_window():
    worker = object.__new__(RecsysBackboneTrainSPMD)
    worker.train_eval_metrics_enabled = False
    worker.world_size = 1

    with pytest.raises(KeyError, match="optimizer_update_window"):
        worker.materialize_train_metrics()


def test_train_core_preserves_engine_normalized_sparse_gradients() -> None:
    class _Executor:
        def prepare_execution(self, inputs, targets):
            del inputs, targets

        def micro_step(self, inputs, targets):
            del inputs, targets
            return (
                TrainStepResult(
                    loss=np.asarray(1.0, dtype=np.float32),
                    predictions={"logits": np.zeros((2, 1), dtype=np.float32)},
                    gradients=None,
                    input_gradients={
                        "group1_dim_16": np.full((2, 1, 16), 12.0, dtype=np.float32)
                    },
                    step=0,
                    micro_step=1,
                    optimizer_applied=False,
                ),
                OptimizerUpdateWindow(
                    micro_step=1,
                    accumulation_steps=3,
                    at_boundary=False,
                    should_apply_update=False,
                    optimizer_step=0,
                ),
            )

    worker = object.__new__(RecsysBackboneTrainSPMD)
    worker._pending_jax_leases = [()]
    worker.sparse_groups = ("group1_dim_16",)
    worker.side_info_groups = ()
    worker.rank = 0
    worker.world_size = 2
    worker._executor = _Executor()
    worker.execution_tracer = None
    worker._last_result = None
    worker._last_update_window = None
    worker._last_update_window_payload = None
    worker._prepared_execution_for_next_batch = False
    worker.micro_step = 0
    worker.grad_accum_steps = 3

    output = worker.train_core(
        jax_input_group1_dim_16=np.zeros((2, 1, 16), dtype=np.float32),
        jax_labels=np.zeros((2,), dtype=np.int32),
        jax_samples_valid_count=np.asarray([2], dtype=np.int32),
    )

    np.testing.assert_allclose(output["grad_jax_group1_dim_16"], 12.0)


def test_ranking_loss_scales_sparse_gradient_once_for_global_batch() -> None:
    import jax
    import jax.numpy as jnp

    def base_loss(logits, labels, row_weights):
        del labels
        return jnp.sum(logits[:, 0] * row_weights)

    adapter = RankingLossAdapter(base_loss)
    logits = jnp.zeros((2, 1), dtype=jnp.float32)
    labels = jnp.zeros((2,), dtype=jnp.int32)
    targets = {
        "labels": labels,
        "samples_valid_count": jnp.asarray([4], dtype=jnp.int32),
        "global_row_start": jnp.asarray(0, dtype=jnp.int32),
        "world_size": jnp.asarray(2, dtype=jnp.int32),
    }
    gradient = jax.grad(lambda values: adapter({"logits": values}, targets))(logits)
    np.testing.assert_allclose(gradient, 0.5)


def test_ranking_loss_masks_global_padding_on_the_later_dp_rank() -> None:
    import jax
    import jax.numpy as jnp

    def base_loss(logits, labels, row_weights):
        del labels
        return jnp.sum(logits[:, 0] * row_weights)

    adapter = RankingLossAdapter(base_loss)
    logits = jnp.zeros((2, 1), dtype=jnp.float32)
    labels = jnp.zeros((2,), dtype=jnp.int32)
    targets = {
        "labels": labels,
        "samples_valid_count": jnp.asarray([3], dtype=jnp.int32),
        "global_row_start": jnp.asarray(2, dtype=jnp.int32),
        "world_size": jnp.asarray(2, dtype=jnp.int32),
    }

    gradient = jax.grad(lambda values: adapter({"logits": values}, targets))(logits)
    np.testing.assert_allclose(
        gradient, np.asarray([[2.0 / 3.0], [0.0]], dtype=np.float32)
    )


def test_update_does_not_capture_audit_without_execution_tracer() -> None:
    class _Executor:
        def __init__(self) -> None:
            self.audit_calls = 0

        def apply_update(self, window):
            del window
            return False

        def capture_audit_state(self):
            self.audit_calls += 1
            raise AssertionError("audit capture must be tracer-only")

        def current_micro_step(self):
            return 0

    worker = object.__new__(RecsysBackboneTrainSPMD)
    worker._last_result = TrainStepResult(
        loss=np.asarray(1.0, dtype=np.float32),
        predictions={},
        gradients=None,
        input_gradients={},
        step=0,
        micro_step=1,
        optimizer_applied=False,
    )
    worker._last_update_window = OptimizerUpdateWindow(
        micro_step=1,
        accumulation_steps=1,
        at_boundary=True,
        should_apply_update=False,
        optimizer_step=0,
    )
    worker._executor = _Executor()
    worker.execution_tracer = None
    worker._last_update_window_payload = None
    worker._last_module_before = None
    worker.micro_step = 0

    update_payload = np.asarray(worker._last_update_window.encode(), dtype=np.int64)
    worker.update(update_payload)

    assert worker._executor.audit_calls == 0


def test_extract_sparse_groups_and_dimensions_numeric_ordering() -> None:
    ae11_input = {
        "hashed_sparse_dim_128": ["feat_128"],
        "hashed_sparse_dim_40": ["feat_40"],
        "hashed_sparse_dim_418": ["feat_418"],
        "hashed_sparse_dim_352": ["feat_352"],
        "hashed_sparse_dim_140": ["feat_140"],
    }
    groups, dims_by_base = extract_sparse_groups_and_dimensions(ae11_input)

    assert groups == (
        "hashed_sparse_dim_128",
        "hashed_sparse_dim_40",
        "hashed_sparse_dim_418",
        "hashed_sparse_dim_352",
        "hashed_sparse_dim_140",
    )
    assert dims_by_base["hashed_sparse"] == [40, 128, 140, 352, 418]

    class _MockStore:
        def table_dims(self):
            return dims_by_base["hashed_sparse"]

        def key_dtype(self):
            return "int64"

        def value_dtype(self):
            return "float32"

        def optimizer_dtype(self):
            return "float32"

        def score_policy(self):
            return "lfu"

    spec = _LanceCheckpointStorage._store_specification(_MockStore())
    assert spec.dimensions == (40, 128, 140, 352, 418)


def test_parse_sparse_group_name_contracts() -> None:
    assert parse_sparse_group_name("table_dim_64") == ("table", 64)
    assert parse_sparse_group_name("user_group_v2_dim_128") == ("user_group_v2", 128)

    with pytest.raises(ValueError, match="non-empty string"):
        parse_sparse_group_name("")
    with pytest.raises(ValueError, match="must contain '_dim_<dimension>'"):
        parse_sparse_group_name("invalid_group_name")
    with pytest.raises(ValueError, match="base name must not be empty"):
        parse_sparse_group_name("_dim_64")
    with pytest.raises(ValueError, match="invalid dimension"):
        parse_sparse_group_name("table_dim_notanint")
    with pytest.raises(ValueError, match="must be positive"):
        parse_sparse_group_name("table_dim_0")
    with pytest.raises(ValueError, match="must be positive"):
        parse_sparse_group_name("table_dim_-10")


def test_parse_sparse_workers_contracts() -> None:
    groups = ("t1_dim_10", "t2_dim_20", "t1_dim_30")
    assert extract_sparse_base_names(groups) == ("t1", "t2")

    assert parse_sparse_workers("", groups) == {"t1": 1, "t2": 1}
    assert parse_sparse_workers("2", groups) == {"t1": 2, "t2": 2}
    assert parse_sparse_workers("t1:3,t2:5", groups) == {"t1": 3, "t2": 5}

    with pytest.raises(ValueError, match="must be an integer"):
        parse_sparse_workers("invalid_single_entry", groups)
    with pytest.raises(ValueError, match="sparse worker override must use name:count"):
        parse_sparse_workers("t1:3,invalid_entry", groups)
    with pytest.raises(ValueError, match="unknown sparse group 'unknown'"):
        parse_sparse_workers("unknown:2", groups)
    with pytest.raises(ValueError, match="duplicate sparse group 't1'"):
        parse_sparse_workers("t1:2,t1:4", groups)
    with pytest.raises(ValueError, match="must be positive"):
        parse_sparse_workers("t1:0", groups)

