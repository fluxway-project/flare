from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pytest
from flax import nnx

from flare.jax.layers import (
    BatchedExpertMoE,
    CategoricalEmbedding,
    fp32_mean,
    fp32_rms_normalize,
    fp32_softmax,
    fp32_sum,
    fp32_weight_gradient_dot_general,
)
from flare.jax.roles import cast_floating_tree, cast_floating_tree_checked
from flare.jax.training import (
    Fp32Param,
    NnxTrainingStateDefinition,
    PrecisionPolicy,
    ReplicatedNnxTrainEngine,
    build_donated_gradient_accumulator,
)
from flare.training.state import TrainingStateLifecycle


class _Model(nnx.Module):
    def __init__(self, param_dtype=jnp.float32):
        self.weight = nnx.Param(jnp.ones((2, 2), dtype=param_dtype))
        self.batch_stat = nnx.BatchStat(jnp.zeros((2,), dtype=jnp.float32))
        self.counter = nnx.Variable(jnp.asarray(1, dtype=jnp.int32))
        self.flag = nnx.Variable(jnp.asarray(True, dtype=jnp.bool_))


class _Factory:
    def __init__(self, param_dtype=jnp.float32):
        self.param_dtype = param_dtype

    def create_module(self):
        return _Model(self.param_dtype)

    def create_optimizer(self, module):
        return nnx.Optimizer(module, optax.adam(0.1), wrt=nnx.Param)


class _Bf16OptimizerFactory(_Factory):
    def create_optimizer(self, module):
        def init_fn(params):
            return jnp.zeros((2, 2), dtype=jnp.bfloat16)

        def update_fn(updates, state, params=None):
            del params
            return updates, state

        return nnx.Optimizer(
            module,
            optax.GradientTransformation(init_fn, update_fn),
            wrt=nnx.Param,
        )


class _IntegerParamFactory(_Factory):
    def create_module(self):
        return _Model(jnp.int32)


class _DenseEmbeddingModel(nnx.Module):
    def __init__(self):
        self.embedding = CategoricalEmbedding(
            {"feature": (3, 4)},
            {"feature": (0, 1)},
            rngs=nnx.Rngs(0),
        )


class _DenseEmbeddingFactory:
    def create_module(self):
        return _DenseEmbeddingModel()

    def create_optimizer(self, module):
        return nnx.Optimizer(module, optax.adam(0.1), wrt=nnx.Param)


def test_policy_has_closed_fixed_dtype_semantics() -> None:
    assert PrecisionPolicy.fp32().compute_dtype == jnp.float32
    assert PrecisionPolicy.fp32().parameter_dtype == jnp.float32
    assert PrecisionPolicy.fp32().optimizer_dtype == jnp.float32
    assert PrecisionPolicy.bf16().compute_dtype == jnp.bfloat16
    assert PrecisionPolicy.bf16().parameter_dtype == jnp.float32
    assert PrecisionPolicy.bf16().optimizer_dtype == jnp.float32
    assert PrecisionPolicy.fp16().compute_dtype == jnp.float16
    assert PrecisionPolicy.fp16().parameter_dtype == jnp.float32
    assert PrecisionPolicy.fp16().optimizer_dtype == jnp.float32
    assert PrecisionPolicy.fp32().sparse_storage_dtype_name == "float32"
    assert PrecisionPolicy.bf16().sparse_storage_dtype_name == "float16"
    assert PrecisionPolicy.fp16().sparse_storage_dtype_name == "float16"
    assert PrecisionPolicy.fp32().sparse_gradient_dtype_name == "float32"
    assert PrecisionPolicy.bf16().sparse_gradient_dtype_name == "float32"
    assert PrecisionPolicy.fp16().sparse_gradient_dtype_name == "float32"
    assert PrecisionPolicy.fp32().sparse_gradient_transport_dtype_name == "float32"
    assert PrecisionPolicy.bf16().sparse_gradient_transport_dtype_name == "bfloat16"
    assert PrecisionPolicy.fp16().sparse_gradient_transport_dtype_name == "bfloat16"
    with pytest.raises(TypeError, match="precision_policy"):
        NnxTrainingStateDefinition.from_factory(_Factory(), precision_policy="bf16")


def test_fp32_weight_gradient_dot_preserves_bf16_forward_and_input_gradient() -> None:
    inputs = jnp.asarray([[1.25, -0.5], [0.75, 2.0]], dtype=jnp.bfloat16)
    kernel = jnp.asarray([[0.5, -0.25], [1.0, 0.75]], dtype=jnp.bfloat16)
    cotangent = jnp.asarray([[0.01, -0.02], [0.03, 0.04]], dtype=jnp.bfloat16)
    dimensions = (((1,), (0,)), ((), ()))

    output, backward = jax.vjp(
        lambda x, weight: fp32_weight_gradient_dot_general(x, weight, dimensions),
        inputs,
        kernel,
    )
    input_gradient, kernel_gradient = backward(cotangent)
    ordinary_output, ordinary_backward = jax.vjp(
        lambda x, weight: jax.lax.dot_general(x, weight, dimensions),
        inputs,
        kernel,
    )
    ordinary_input_gradient, _ = ordinary_backward(cotangent)
    expected_kernel_gradient = (
        inputs.astype(jnp.float32).T @ cotangent.astype(jnp.float32)
    ).astype(jnp.bfloat16)

    np.testing.assert_array_equal(output, ordinary_output)
    np.testing.assert_array_equal(input_gradient, ordinary_input_gradient)
    np.testing.assert_array_equal(kernel_gradient, expected_kernel_gradient)
    with pytest.raises(ValueError, match="dense linear dimensions"):
        fp32_weight_gradient_dot_general(inputs, kernel, (((0,), (0,)), ((), ())))


def test_donated_gradient_accumulator_aliases_accumulator() -> None:
    accumulator = build_donated_gradient_accumulator()
    lowered = accumulator.lower(
        (jnp.ones((8,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32)),
        (jnp.ones((8,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32)),
    )
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    assert "input_output_alias=" in hlo
    assert "(0, {}, may-alias)" in hlo
    result = accumulator(
        (jnp.ones((8,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32)),
        (
            jnp.full((8,), 2.0, dtype=jnp.float32),
            jnp.full((4,), 3.0, dtype=jnp.float32),
        ),
    )
    assert jnp.all(result[0] == 3.0)
    assert jnp.all(result[1] == 4.0)


def test_distributed_bf16_runtime_contract() -> None:
    probe = Path(__file__).with_name("bf16_distributed_runtime_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "BF16_DISTRIBUTED_RUNTIME=PASS" in result.stdout


def test_dense_embedding_stays_fp32_with_bf16_policy() -> None:
    factory = _DenseEmbeddingFactory()
    fp32_definition = NnxTrainingStateDefinition.from_factory(factory)
    fp32_model = fp32_definition.initialize_values()["model"]
    fp32_compute = fp32_definition.prepare_compute_module(fp32_model)
    assert fp32_model.embedding.emb_4.get_value().dtype == jnp.float32
    assert fp32_compute.embedding.emb_4.get_value().dtype == jnp.float32
    assert fp32_compute is fp32_model
    assert (
        fp32_compute.embedding(jnp.asarray([[1]], dtype=jnp.int32))[4].dtype
        == jnp.float32
    )

    bf16_definition = NnxTrainingStateDefinition.from_factory(
        factory, precision_policy=PrecisionPolicy.bf16()
    )
    bf16_model = bf16_definition.initialize_values()["model"]
    bf16_compute = bf16_definition.prepare_compute_module(bf16_model)
    assert bf16_model.embedding.emb_4.get_value().dtype == jnp.float32
    assert isinstance(bf16_compute.embedding.emb_4, Fp32Param)
    assert bf16_compute.embedding.emb_4.get_value().dtype == jnp.float32
    assert (
        bf16_compute.embedding(jnp.asarray([[1]], dtype=jnp.int32))[4].dtype
        == jnp.float32
    )


def test_embedding_output_boundary_cast_is_policy_explicit() -> None:
    embedding = CategoricalEmbedding(
        {"feature": (3, 4)},
        {"feature": (0, 1)},
        rngs=nnx.Rngs(0),
    )
    outputs = embedding(jnp.asarray([[1]], dtype=jnp.int32))
    assert outputs[4].dtype == jnp.float32
    assert (
        CategoricalEmbedding.cast_outputs(outputs, jnp.float32)[4].dtype == jnp.float32
    )
    assert (
        CategoricalEmbedding.cast_outputs(outputs, jnp.bfloat16)[4].dtype
        == jnp.bfloat16
    )

    class _DenseInputModule(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))

    module = _DenseInputModule()
    assert CategoricalEmbedding.infer_compute_dtype(module) == jnp.float32

    class _MixedRoleModule(nnx.Module):
        def __init__(self):
            self.compute_weight = nnx.Param(jnp.ones((2, 2), dtype=jnp.bfloat16))
            self.embedding_weight = Fp32Param(jnp.ones((2, 2), dtype=jnp.float32))

    assert CategoricalEmbedding.infer_compute_dtype(_MixedRoleModule()) == jnp.bfloat16


def test_bf16_embedding_pooling_boundary_feeds_bf16_gemm() -> None:
    class _DenseEmbeddingLinear(nnx.Module):
        def __init__(self):
            self.embedding = CategoricalEmbedding(
                {"feature": (3, 4)}, {"feature": (0, 2)}, rngs=nnx.Rngs(0)
            )
            self.projection = nnx.Linear(4, 2, rngs=nnx.Rngs(1))

        def __call__(self, inputs):
            pooled_fp32 = self.embedding(inputs["side_info"])[4]
            compute_dtype = CategoricalEmbedding.infer_compute_dtype(self)
            pooled_compute = CategoricalEmbedding.cast_outputs(
                {4: pooled_fp32}, compute_dtype
            )[4]
            return self.projection(pooled_compute)

    class _DenseEmbeddingLinearFactory:
        def create_module(self):
            return _DenseEmbeddingLinear()

        def create_optimizer(self, module):
            return nnx.Optimizer(module, optax.adam(0.1), wrt=nnx.Param)

    definition = NnxTrainingStateDefinition.from_factory(
        _DenseEmbeddingLinearFactory(), precision_policy=PrecisionPolicy.bf16()
    )
    model = definition.initialize_values()["model"]
    compute = definition.prepare_compute_module(model)
    inputs = {"side_info": jnp.asarray([[1, 2]], dtype=jnp.int32)}
    output = compute(inputs)
    assert output.dtype == jnp.bfloat16
    jaxpr = jax.make_jaxpr(compute)(inputs)
    assert "bf16" in str(jaxpr)
    assert "dot_general" in str(jaxpr)


def test_dense_embedding_initializer_stays_fp32_with_x64_enabled() -> None:
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        embedding = CategoricalEmbedding(
            {"feature": (3, 4)},
            {"feature": (0, 1)},
            rngs=nnx.Rngs(0),
        )
        assert embedding.emb_4.get_value().dtype == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_dense_embedding_engine_keeps_fp32_compute_dtype() -> None:
    seen = []

    def forward(module, inputs):
        embeddings = module.embedding(inputs["side_info"])[4]
        seen.append((module.embedding.emb_4.get_value().dtype, embeddings.dtype))
        return {"logits": embeddings}

    def loss(predictions, targets):
        return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)

    inputs = {"side_info": jnp.asarray([[1], [2]], dtype=jnp.int32)}
    targets = {"labels": jnp.zeros((2, 1, 4), dtype=jnp.float32)}
    for policy in (PrecisionPolicy.fp32(), PrecisionPolicy.bf16()):
        definition = NnxTrainingStateDefinition.from_factory(
            _DenseEmbeddingFactory(), precision_policy=policy
        )
        engine = ReplicatedNnxTrainEngine(definition, forward, loss)
        engine.initialize()
        result = engine.train_step(inputs, targets)
        assert result.loss.dtype == jnp.float32
        assert result.predictions["logits"].dtype == jnp.float32
        assert seen[-1] == (jnp.float32, jnp.float32)
        assert engine.module.embedding.emb_4.get_value().dtype == jnp.float32


def test_dense_embedding_param_role_is_strict_across_checkpoint_restore() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _DenseEmbeddingFactory(), precision_policy=PrecisionPolicy.bf16()
    )
    source = ReplicatedNnxTrainEngine(
        definition,
        lambda module, inputs: {"logits": module.embedding(inputs["side_info"])[4]},
        lambda predictions, targets: jnp.mean(
            (predictions["logits"] - targets["labels"]) ** 2
        ),
    )
    source.initialize()
    checkpoint = source.capture_checkpoint_state()
    restored = ReplicatedNnxTrainEngine(
        NnxTrainingStateDefinition.from_factory(_DenseEmbeddingFactory()),
        source.forward_fn,
        source.loss_fn,
        state_lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
    )
    restored.restore_checkpoint_state(checkpoint)
    assert isinstance(restored.module.embedding.emb_4, Fp32Param)

    incompatible = _DenseEmbeddingModel()
    incompatible.embedding.emb_4 = nnx.Param(incompatible.embedding.emb_4.get_value())
    _, incompatible_state = nnx.split(incompatible)
    with pytest.raises(ValueError, match="state tree differs"):
        definition.initialize_from_model_state(incompatible_state)


def test_definition_normalizes_master_and_preserves_non_param_state() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(jnp.float16), precision_policy=PrecisionPolicy.bf16()
    )
    values = definition.initialize_values()
    model = values["model"]
    assert model.weight.get_value().dtype == jnp.float32


def test_compute_view_fast_path_skips_static_state_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    model = definition.initialize_values()["model"]

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("static state validation must be preflight-only")

    monkeypatch.setattr(definition.model_definition, "validate_state", fail_validation)
    compute = definition.prepare_compute_module(model)
    assert compute.weight.get_value().dtype == jnp.bfloat16
    assert model.batch_stat.get_value().dtype == jnp.float32
    assert model.counter.get_value().dtype == jnp.int32
    assert model.flag.get_value().dtype == jnp.bool_
    compute = definition.prepare_compute_module(model)
    assert compute.weight.get_value().dtype == jnp.bfloat16
    assert compute.batch_stat.get_value().dtype == jnp.float32
    assert compute.counter.get_value().dtype == jnp.int32
    assert compute.flag.get_value().dtype == jnp.bool_
    compute.batch_stat[...] = compute.batch_stat[...] + 1
    definition.sync_non_param_state(model, compute)
    np.testing.assert_array_equal(model.batch_stat.get_value(), 1.0)
    assert model.weight.get_value().dtype == jnp.float32


def test_fp16_compute_view_and_input_boundary() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.fp16()
    )
    model = definition.initialize_values()["model"]
    compute = definition.prepare_compute_module(model)
    assert model.weight.get_value().dtype == jnp.float32
    assert compute.weight.get_value().dtype == jnp.float16
    prepared = definition.prepare_inputs({"dense": jnp.ones((2, 2), dtype=jnp.float32)})
    assert prepared["dense"].dtype == jnp.float16


def test_bf16_sparse_input_crosses_compute_boundary() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    sparse_lookup = jnp.asarray([[[1.5, -2.0]]], dtype=jnp.float16)
    prepared = definition.prepare_inputs({"sparse_group": sparse_lookup})
    assert prepared["sparse_group"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        prepared["sparse_group"], sparse_lookup.astype(jnp.bfloat16)
    )


def test_fp16_input_cast_rejects_nonfinite_and_out_of_range_values() -> None:
    finite = cast_floating_tree_checked(
        {"dense": jnp.asarray([1.5, -2.0], dtype=jnp.float32)},
        jnp.float16,
        boundary="test input",
    )
    assert finite["dense"].dtype == jnp.float16
    with pytest.raises(FloatingPointError, match="test input"):
        cast_floating_tree_checked(
            {"dense": jnp.asarray([jnp.inf], dtype=jnp.float32)},
            jnp.float16,
            boundary="test input",
        )
    with pytest.raises(FloatingPointError, match="test input"):
        cast_floating_tree_checked(
            {"dense": jnp.asarray([65505.0], dtype=jnp.float32)},
            jnp.float16,
            boundary="test input",
        )
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.fp16()
    )
    prepared = definition.prepare_inputs(
        {"dense": jnp.asarray([65505.0], dtype=jnp.float32)}
    )
    assert prepared["dense"].dtype == jnp.float16

    @jax.jit
    def cast_jitted(inputs):
        return cast_floating_tree(inputs, jnp.float16)

    jitted_result = cast_jitted({"dense": jnp.asarray([1.5, -2.0], dtype=jnp.float32)})[
        "dense"
    ]
    jitted_result.block_until_ready()
    assert jitted_result.dtype == jnp.float16
    np.testing.assert_array_equal(jitted_result, jnp.asarray([1.5, -2.0], jnp.float16))


def test_fp16_sparse_input_gradient_crosses_fp32_update_boundary() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.fp16()
    )
    sparse_gradient = jnp.asarray([[[1.5, -2.0]]], dtype=jnp.float16)
    cast_gradient = definition.cast_gradients({"sparse_group": sparse_gradient})
    assert cast_gradient["sparse_group"].dtype == jnp.float32
    np.testing.assert_array_equal(
        cast_gradient["sparse_group"], sparse_gradient.astype(jnp.float32)
    )


def test_constant_diagnostic_skips_host_callback_without_constant_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    model = definition.initialize_values()["model"]
    compute = definition.prepare_compute_module(model)

    def fail_callback(*_args, **_kwargs):
        raise AssertionError("empty ConstantBuffer state must not register a callback")

    monkeypatch.setattr(jax.debug, "callback", fail_callback)
    monkeypatch.setattr("flare.jax.training.nnx.state.nnx.update", lambda *_args: None)
    jax.make_jaxpr(
        lambda: definition.sync_non_param_state(
            model, compute, validate_constant_values=True
        )
    )()


def test_compute_view_jaxpr_contains_bf16_dot() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    model = definition.initialize_values()["model"]
    compute = definition.prepare_compute_module(model)
    jaxpr = jax.make_jaxpr(lambda x: jnp.sum(x @ compute.weight.get_value()))(
        jnp.ones((2, 2), dtype=jnp.bfloat16)
    )
    dot_eqns = [
        eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive is jax.lax.dot_general_p
    ]
    assert dot_eqns
    assert all(
        getattr(variable, "aval", variable).dtype == jnp.bfloat16
        for variable in dot_eqns[0].invars
    )


def test_external_bf16_state_becomes_fp32_master() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    external = _Model(jnp.bfloat16)
    _, external_state = nnx.split(external)
    model = definition.materialize_model_from_state(external_state)
    assert model.weight.get_value().dtype == jnp.float32
    assert model.batch_stat.get_value().dtype == jnp.float32
    assert model.counter.get_value().dtype == jnp.int32


@pytest.mark.parametrize("source_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_external_allowed_param_dtypes_become_fp32_master(source_dtype) -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    external = _Model(source_dtype)
    _, external_state = nnx.split(external)
    model = definition.materialize_model_from_state(external_state)
    assert model.weight.get_value().dtype == jnp.float32


def test_fp32_external_non_master_param_dtype_is_rejected() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.fp32()
    )
    external = _Model(jnp.bfloat16)
    _, external_state = nnx.split(external)
    with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
        definition.materialize_model_from_state(external_state)


def test_non_floating_param_is_rejected() -> None:
    with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
        NnxTrainingStateDefinition.from_factory(
            _IntegerParamFactory(), precision_policy=PrecisionPolicy.bf16()
        )


def test_bfloat16_optimizer_state_is_rejected() -> None:
    with pytest.raises(TypeError, match="optimizer floating state"):
        NnxTrainingStateDefinition.from_factory(
            _Bf16OptimizerFactory(), precision_policy=PrecisionPolicy.bf16()
        )


def test_model_state_shape_and_non_param_dtype_conflicts_fail() -> None:
    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    _, state = nnx.split(_Model())
    bad_shape = jtu.tree_map_with_path(
        lambda path, leaf: (
            jnp.ones((3, 2), dtype=jnp.bfloat16)
            if jtu.keystr(path) == "['weight'].value"
            else leaf
        ),
        state,
    )
    with pytest.raises(ValueError, match="shape differs"):
        definition.materialize_model_from_state(bad_shape)
    bad_non_param = jtu.tree_map_with_path(
        lambda path, leaf: (
            jnp.zeros((2,), dtype=jnp.float16)
            if jtu.keystr(path) == "['batch_stat'].value"
            else leaf
        ),
        state,
    )
    with pytest.raises(ValueError, match="dtype differs"):
        definition.materialize_model_from_state(bad_non_param)


def test_engine_keeps_loss_gradients_and_optimizer_boundary_fp32() -> None:
    seen = []

    def forward(module, inputs):
        seen.append((module.weight.get_value().dtype, inputs["x"].dtype))
        module.batch_stat[...] = module.batch_stat[...] + 1
        return {"logits": inputs["x"] @ module.weight.get_value()}

    def loss(predictions, targets):
        return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)

    definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    engine = ReplicatedNnxTrainEngine(definition, forward, loss, grad_accum_steps=2)
    engine.initialize()
    batch = (
        {"x": jnp.ones((2, 2), dtype=jnp.float32)},
        {"labels": jnp.zeros((2, 2), dtype=jnp.float32)},
    )
    result = engine.train_micro_step(*batch)
    assert result.loss.dtype == jnp.float32
    assert result.predictions["logits"].dtype == jnp.float32
    assert result.gradients.weight.get_value().dtype == jnp.float32
    assert result.input_gradients["x"].dtype == jnp.float32
    assert engine.module.weight.get_value().dtype == jnp.float32
    engine.train_step(*batch)
    _, optimizer_state = nnx.split(engine.optimizer)
    assert all(
        not jnp.issubdtype(leaf.dtype, jnp.floating) or leaf.dtype == jnp.float32
        for leaf in jtu.tree_leaves(optimizer_state)
    )
    assert seen == [(jnp.bfloat16, jnp.bfloat16), (jnp.bfloat16, jnp.bfloat16)]


def test_engine_differentiates_only_declared_sparse_inputs() -> None:
    def forward(module, inputs):
        return {"logits": inputs["x"] @ module.weight.get_value() + inputs["side"]}

    def loss(predictions, targets):
        return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)

    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    engine = ReplicatedNnxTrainEngine(
        definition,
        forward,
        loss,
        input_gradient_keys=("x",),
    )
    engine.initialize()
    result = engine.train_micro_step(
        {
            "x": jnp.ones((2, 2), dtype=jnp.float32),
            "side": jnp.ones((2, 2), dtype=jnp.float32),
        },
        {"labels": jnp.zeros((2, 2), dtype=jnp.float32)},
    )
    assert set(result.input_gradients) == {"x"}
    assert result.input_gradients["x"].shape == (2, 2)


def test_float64_param_is_rejected_when_x64_is_enabled() -> None:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
            NnxTrainingStateDefinition.from_factory(
                _Factory(jnp.float64), precision_policy=PrecisionPolicy.bf16()
            )
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_checkpoint_cross_restores_between_fp32_and_bf16_runtime() -> None:
    def forward(module, inputs):
        return {"logits": inputs["x"] @ module.weight.get_value()}

    def loss(predictions, targets):
        return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)

    fp32_definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.fp32()
    )
    bf16_definition = NnxTrainingStateDefinition.from_factory(
        _Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    batch = (
        {"x": jnp.ones((2, 2), dtype=jnp.float32)},
        {"labels": jnp.zeros((2, 2), dtype=jnp.float32)},
    )
    fp32_engine = ReplicatedNnxTrainEngine(fp32_definition, forward, loss)
    fp32_engine.initialize()
    fp32_engine.train_step(*batch)
    fp32_checkpoint = fp32_engine.capture_checkpoint_state()
    bf16_engine = ReplicatedNnxTrainEngine(
        bf16_definition,
        forward,
        loss,
        state_lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
    )
    bf16_engine.restore_checkpoint_state(fp32_checkpoint)
    assert bf16_engine.module.weight.get_value().dtype == jnp.float32

    bf16_source = ReplicatedNnxTrainEngine(bf16_definition, forward, loss)
    bf16_source.initialize()
    bf16_source.train_step(*batch)
    bf16_checkpoint = bf16_source.capture_checkpoint_state()
    fp32_restore = ReplicatedNnxTrainEngine(
        fp32_definition,
        forward,
        loss,
        state_lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
    )
    fp32_restore.restore_checkpoint_state(bf16_checkpoint)
    assert fp32_restore.module.weight.get_value().dtype == jnp.float32


@pytest.mark.parametrize(
    "invalid_step",
    [
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([1], dtype=jnp.int32),
    ],
)
def test_checkpoint_rejects_invalid_step_contract(invalid_step) -> None:
    definition = NnxTrainingStateDefinition.from_factory(_Factory())
    source = ReplicatedNnxTrainEngine(
        definition,
        lambda module, inputs: {"logits": inputs["x"] @ module.weight.get_value()},
        lambda predictions, targets: jnp.mean(
            (predictions["logits"] - targets["labels"]) ** 2
        ),
    )
    source.initialize()
    checkpoint = source.capture_checkpoint_state()
    checkpoint["step"] = invalid_step
    restored = ReplicatedNnxTrainEngine(
        definition,
        source.forward_fn,
        source.loss_fn,
        state_lifecycle=TrainingStateLifecycle.CHECKPOINT_RESTORE,
    )
    with pytest.raises((TypeError, ValueError), match="checkpoint step"):
        restored.restore_checkpoint_state(checkpoint)
    assert restored.module is None
    assert restored.optimizer is None
    assert restored.step == 0


def test_sensitive_reductions_compute_in_fp32_and_restore_dtype() -> None:
    values = jnp.asarray([[1.0, 2.0]], dtype=jnp.bfloat16)
    probabilities = fp32_softmax(values)
    average = fp32_mean(values, axis=-1)
    normalized = fp32_rms_normalize(values)
    summed = fp32_sum(values, axis=-1)

    assert probabilities.dtype == jnp.bfloat16
    assert average.dtype == jnp.bfloat16
    assert normalized.dtype == jnp.bfloat16
    assert summed.dtype == jnp.bfloat16
    assert jnp.allclose(
        jnp.sum(probabilities.astype(jnp.float32), axis=-1),
        jnp.ones((1,), dtype=jnp.float32),
    )
    softmax_jaxpr = str(jax.make_jaxpr(fp32_softmax)(values))
    assert "convert_element_type" in softmax_jaxpr


def test_moe_sensitive_path_accepts_bf16_compute_inputs() -> None:
    class Factory:
        def create_module(self):
            return BatchedExpertMoE(
                batch_dim=1,
                in_dim=4,
                scale_size=1,
                num_experts=2,
                sparse=False,
                top_k=2,
                rngs=nnx.Rngs(0),
            )

        def create_optimizer(self, module):
            return nnx.Optimizer(module, optax.sgd(0.1), wrt=nnx.Param)

    definition = NnxTrainingStateDefinition.from_factory(
        Factory(), precision_policy=PrecisionPolicy.bf16()
    )
    module = definition.prepare_compute_module(definition.initialize_values()["model"])
    output = module(jnp.ones((1, 2, 4), dtype=jnp.bfloat16))

    assert output.dtype == jnp.bfloat16
