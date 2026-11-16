from types import SimpleNamespace

import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from fluxway.common.types.checkpoint import CheckpointRetentionPolicy
from fluxway.control_plane.graph.functional import FunctionalGraph
from sparse_embedding.checkpoint import PartitionSpec

from flare.data.pipeline.consumption.bootstrap.factory import STANDARD_SOURCE_FACTORY
from flare.data.schema.feature_catalog import (
    FeatureCatalog,
    FeatureDescriptor,
    FeatureOutputKind,
    FeatureRepresentation,
)
from flare.jax.layers.state import NnxModelDefinition
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.precision import PrecisionPolicy
from flare.mesh import build_1d_device_mesh
from flare.training.components.sparse_table.contracts import SparseOptimizerConfig
from flare.training.execution import TrainingExecutionPlan
from recipes.recsys.ranking.graph.bindings import (
    bind_recsys_ranking_evaluation,
    bind_recsys_ranking_metrics,
    bind_recsys_ranking_training,
)
from recipes.recsys.ranking.graph.forward import ranking_forward
from recipes.recsys.ranking.graph.losses import RankingLossAdapter
from recipes.recsys.ranking.graph.pipeline import (
    build_pipeline_inputs,
    build_pipeline_nodes,
)
from recipes.recsys.ranking.losses import MultiTargetCrossEntropyLoss
from recipes.recsys.ranking.task import RankingTask
from recipes.recsys.ranking.topology import (
    assemble_ranking_evaluation_graph,
    assemble_ranking_training_graph,
)


class Model(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.ones((2, 1)))

    def __call__(self, inputs):
        return {"logits": jnp.sum(inputs["entity_dim_2"], axis=1) @ self.weight}


def assert_native_graph_assembly(tmp_path, mode, label, group, config_factory):
    descriptors = [
        FeatureDescriptor(
            code=code,
            name=name,
            representation=FeatureRepresentation.FID,
            length=1,
            output_kind=FeatureOutputKind.SPARSE_I64_FIDS,
            encoder="Identity",
            cross_type=None,
            cross_class=None,
            dependency_codes=(),
        )
        for code, name in [(1, "entity"), (2, group)]
    ]
    FeatureCatalog(
        descriptors,
        config_digest="0" * 64,
        columns=("fid_1", "fid_2", label),
    ).write_manifest(tmp_path)
    args = config_factory(
        reader_workers=2,
        dp=1,
        sparse_workers="1",
        jax_platform="cpu",
        sparse_max_staleness_steps=1,
        global_batch_size=4,
        completion_policy="drop",
        grad_accum_steps=1,
        source_assignment_policy="none",
        sparse_ckpt_backend=None,
        precision=PrecisionPolicy.FP32,
        feature_manifest_dir=str(tmp_path),
        null_index_sentinel=-1,
        source_read_concurrency=1,
        dataset_path=str(tmp_path / "dataset"),
        storage_format="parquet",
        file_storage_options={},
        catalog_name=None,
        catalog_type=None,
        catalog_uri=None,
        warehouse=None,
        iceberg_config=None,
        iceberg_snapshot_id=None,
        table_identifier=None,
        iceberg_max_open_files=None,
        reader_execution_audit_path=None,
        sparse_reserve_capacity=0,
        sparse_optimizer=SparseOptimizerConfig(),
        validate_tensor_values=False,
    )
    builder = SimpleNamespace(
        pack_meta=SimpleNamespace(
            sparse_id_groups={"entity_dim_2": ["entity"]}, side_info_seq=()
        ),
    )
    task = RankingTask(
        label,
        group,
        MultiTargetCrossEntropyLoss.create(
            label_map={"negative": 0, "positive": 1},
            target_names=("engagement",),
            sample_weight_map={
                "engagement": {"sample_weight": {"negative": -1, "positive": 1}}
            },
        ),
    )
    common = build_pipeline_nodes(
        builder,
        args,
        source_factory=STANDARD_SOURCE_FACTORY,
        manifest_source_enabled=True,
        optimizer_enabled=mode == "train",
        load_optimizer_state=False,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        sparse_partition_assignment_spec=PartitionSpec.modulo(),
        key_dtype="int64",
    )
    definition = NnxModelDefinition.from_initializer(Model)
    execution_plan = TrainingExecutionPlan.replicated()
    binding_options = {
        "sparse_groups": ("entity_dim_2",),
        "side_info_groups": ("side_info",),
        "resource_profile": common.resources.get_role("compute"),
        "runtime_env": {},
        "dp_size": 1,
    }
    compute = (
        bind_recsys_ranking_training(
            **binding_options, pipeline_depth=2, execution_plan=execution_plan
        )
        if mode == "train"
        else bind_recsys_ranking_evaluation(
            **binding_options, model_definition=definition
        )
    )
    mesh = build_1d_device_mesh(1, axis_name="dp")
    with FunctionalGraph("synthetic-ranking") as graph:
        inputs = build_pipeline_inputs(
            mb=builder,
            side_info_seq=(),
            args=args,
            catalog_token=None,
            common=common,
            sparse_workers_config={"entity": 1},
            repeat_first_batch=False,
            ranking_task=task,
        )
        if mode == "train":
            state = NnxTrainingStateDefinition.from_model_definition(
                definition,
                lambda model: nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param),
            )
            result = assemble_ranking_training_graph(
                graph,
                graph_inputs=inputs,
                training_spmd=compute,
                training_mesh=mesh,
                training_static_args={
                    "state_definition": state,
                    "sparse_groups": ("entity_dim_2",),
                    "side_info_groups": ("side_info",),
                    "sparse_group_widths": {"entity_dim_2": (1, 1)},
                    "sparse_tensor_backend": "jax",
                    "forward_fn": ranking_forward,
                    "loss_fn": RankingLossAdapter(task.objective),
                    "execution_plan": execution_plan,
                },
                placement_recipe_id="synthetic",
                compute_spmd_name="Compute",
                metrics_spmd_name="Metrics",
            )
        else:
            metrics = bind_recsys_ranking_metrics(
                input_sharding={
                    "scores": {"type": "static", "axis": ["dp"]},
                    "labels": {"type": "static", "axis": ["dp"]},
                    "group_ids": {"type": "static", "axis": ["dp"]},
                    "samples_valid_count": {"type": "static", "axis": []},
                },
                resource_profile=common.resources.get_role("metrics"),
                runtime_env={},
            )
            result = assemble_ranking_evaluation_graph(
                graph,
                graph_inputs=inputs,
                evaluation_spmd=compute,
                evaluation_mesh=mesh,
                evaluation_static_args={
                    "model_definition": definition,
                    "sparse_groups": ("entity_dim_2",),
                    "side_info_groups": ("side_info",),
                    "sparse_group_widths": {"entity_dim_2": (1, 1)},
                    "sparse_tensor_backend": "jax",
                    "forward_fn": ranking_forward,
                    "RankingOutput": common.data_types.ranking_eval_output,
                    "weight_source": object(),
                    "sparse_embedding_dtype": "float32",
                },
                metrics_spmd=metrics,
                metrics_mesh=mesh,
                metrics_static_args={
                    "target_names": task.objective.targets,
                    "target_labels": task.objective.target_labels,
                },
                placement_recipe_id="synthetic",
                compute_spmd_name="Compute",
                metrics_spmd_name="Metrics",
            )
    compiled = result["task_graph"].require_compiled_graph()
    if mode == "train":
        # All remote sparse-update traffic, including the window token, must
        # originate at scatter. It must not consume train_core's send credits.
        update_edges = [
            edge.spec
            for edge in compiled.dependencies
            if edge.spec.target.spmd_id.startswith("Embedding_")
            and edge.spec.target.op_id == "update"
        ]
        assert update_edges
        assert all(edge.source.op_id == "scatter_sparse_grads" for edge in update_edges)
        dense_update_edges = [
            edge.spec
            for edge in compiled.dependencies
            if edge.spec.source.op_id == "train_core"
            and edge.spec.target.op_id == "update"
        ]
        assert len(dense_update_edges) == 1
        assert dense_update_edges[0].condition is not None
        prepare_bind = [
            edge.spec
            for edge in compiled.dependencies
            if edge.spec.source.op_id == "prepare_sparse_materialize"
            and edge.spec.target.op_id == "prepare_jax_bind"
        ]
        assert prepare_bind
        assert all(
            transfer.target.name != "padded_inverse_entity_dim_2"
            for edge in prepare_bind
            for transfer in edge.transfers
        )
        prepare_scatter = [
            edge.spec
            for edge in compiled.dependencies
            if edge.spec.source.op_id == "prepare_sparse_materialize"
            and edge.spec.target.op_id == "scatter_sparse_grads"
        ]
        assert any(
            transfer.target.name == "padded_inverse_entity_dim_2"
            for edge in prepare_scatter
            for transfer in edge.transfers
        )
    assert (
        compiled.spmd("Reader").operator("read").execution_layout.logical_mesh.size == 1
    )
    assert result["placement"].recipe_id == "synthetic"
    reader_config = (
        result["task_graph"].nodes["Reader"].spmd_config.static_args["reader_config"]
    )
    assert reader_config.group_id_column == group
    assert set(reader_config.feature_layout.projection) == {"fid_1", "fid_2", label}
    return args


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize(
    "label,group", [("outcome", "visitor"), ("feedback", "session")]
)
def test_native_graph_assembly_uses_injected_task_and_source(
    tmp_path, mode, label, group
):
    assert_native_graph_assembly(tmp_path, mode, label, group, SimpleNamespace)


def test_training_phase_gates_match_ranking_causal_schedule():
    from fluxway.common.schedule_semantics import compile_phase_schedule

    from recipes.recsys.ranking.workers.schedule import (
        RANKING_PHASE_LANES,
        ranking_phase_schedule,
    )

    policies = {phase: ranking_phase_schedule(phase) for phase in RANKING_PHASE_LANES}
    gates = compile_phase_schedule(policies).gates
    assert gates["prepare_jax_bind"].predecessors == (
        ("prepare_sparse_materialize", 0),
    )
    assert gates["train_core"].predecessors == (
        ("prepare_jax_bind", 0),
        ("update", 1),
    )
    assert gates["scatter_sparse_grads"].predecessors == (("train_core", 0),)


def test_dense_update_condition_skips_non_boundary_windows():
    from types import SimpleNamespace

    import numpy as np
    from fluxway.spmd.worker.operator_manager.dispatcher import (
        ExecutionResult,
        ExecutionStatus,
    )

    from flare.training.execution import OptimizerUpdateWindow
    from recipes.recsys.ranking.topology import _should_apply_dense_update

    non_boundary = OptimizerUpdateWindow(
        micro_step=1,
        accumulation_steps=3,
        at_boundary=False,
        should_apply_update=False,
        optimizer_step=0,
    )
    boundary = OptimizerUpdateWindow(
        micro_step=3,
        accumulation_steps=3,
        at_boundary=True,
        should_apply_update=True,
        optimizer_step=0,
    )

    assert not _should_apply_dense_update(
        ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            data={"optimizer_update_window": np.asarray(non_boundary.encode())},
        )
    )
    assert _should_apply_dense_update(
        SimpleNamespace(data={"optimizer_update_window": np.asarray(boundary.encode())})
    )


def test_sparse_owner_cpu_override_reaches_bound_table(tmp_path, monkeypatch):
    from recipes.recsys.ranking.graph import pipeline

    original = pipeline.get_sparse_table_class
    observed = []

    def capture(*args, **kwargs):
        observed.append(args[3].base["CPU"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "get_sparse_table_class", capture)
    assert_native_graph_assembly(
        tmp_path,
        "train",
        "outcome",
        "visitor",
        lambda **kwargs: SimpleNamespace(**kwargs, sparse_table_cpus="entity:2"),
    )
    assert observed and all(value == 2 for value in observed)


def test_sparse_owner_thread_budget_is_independent_of_ray_cpus(tmp_path, monkeypatch):
    from recipes.recsys.ranking.graph import pipeline

    original = pipeline.get_sparse_table_class
    observed = []

    def capture(*args, **kwargs):
        bound = original(*args, **kwargs)
        observed.append((args[3].base["CPU"], bound._factory_thread_budget))
        return bound

    monkeypatch.setattr(pipeline, "get_sparse_table_class", capture)
    assert_native_graph_assembly(
        tmp_path,
        "train",
        "outcome",
        "visitor",
        lambda **kwargs: SimpleNamespace(
            **kwargs,
            sparse_table_cpus="entity:2",
            sparse_table_threads="entity:64",
        ),
    )
    assert observed and all(cpu == 2 and threads == 64 for cpu, threads in observed)
