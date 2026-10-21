from collections.abc import Sequence

import jax
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.layers.state import NnxModelDefinition
from flare.jax.mesh import ParallelMesh
from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.optimizers import create_nnx_optimizer
from flare.jax.training.steppers import require_distributed_execution
from flare.jax.training.strategy import (
    CheckpointShardingStrategy,
    DataParallelStrategy,
    ZeroShardingStrategy,
)
from flare.training.execution.plan import (
    GradientAccumulationPolicy,
    MacroUpdatePolicy,
    TrainingExecutionPlan,
    ZeroExecutionPolicy,
)
from flare.training.zero.contracts import (
    ZeroPartitionContract,
    ZeroStage,
)


class _MockDevice:
    def __init__(self, id: int) -> None:
        self.id = id

    def __repr__(self) -> str:
        return f"_MockDevice({self.id})"


class SimpleModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.w1 = nnx.Param(jax.random.normal(rngs.params(), (8, 16)))
        self.b1 = nnx.Param(jax.random.normal(rngs.params(), (16,)))
        self.w2 = nnx.Param(jax.random.normal(rngs.params(), (16, 4)))
        self.b2 = nnx.Param(jax.random.normal(rngs.params(), (4,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w1[...] + self.b1[...]


def _mock_devices(count: int) -> Sequence[_MockDevice]:
    return tuple(_MockDevice(i) for i in range(count))


def test_parallel_mesh_from_axes_success() -> None:
    devices = _mock_devices(4)
    mesh = ParallelMesh.from_axes(devices=devices, data=2, fsdp=2)
    assert mesh.size == 4
    assert mesh.axis_names == ("data", "fsdp")
    assert mesh["data"] == 2
    assert mesh["fsdp"] == 2
    assert mesh.axis_size("data") == 2
    assert mesh.shape["data"] == 2
    assert len(mesh.devices) == 4
    assert isinstance(mesh.mesh, Mesh)


def test_parallel_mesh_from_axes_validation_errors() -> None:
    devices = _mock_devices(4)
    with pytest.raises(ValueError, match="at least one axis must be specified"):
        ParallelMesh.from_axes(devices=devices)
    with pytest.raises(ValueError, match="axis 'data' size must be positive"):
        ParallelMesh.from_axes(devices=devices, data=0)
    with pytest.raises(ValueError, match="axis name must be non-empty"):
        ParallelMesh.from_axes(devices=devices, **{"": 4})
    with pytest.raises(
        ValueError, match="device count 4 does not match axes product 2"
    ):
        ParallelMesh.from_axes(devices=devices, data=2)
    with pytest.raises(ValueError, match="devices must not be empty"):
        ParallelMesh.from_axes(devices=(), data=1)


def test_parallel_mesh_from_dp() -> None:
    devices = _mock_devices(4)
    mesh = ParallelMesh.from_dp(devices=devices, axis_name="dp")
    assert mesh.size == 4
    assert mesh.axis_names == ("dp",)
    assert mesh["dp"] == 4

    sub_mesh = ParallelMesh.from_dp(size=2, devices=devices, axis_name="data")
    assert sub_mesh.size == 2
    assert sub_mesh["data"] == 2

    with pytest.raises(ValueError, match="size must be positive"):
        ParallelMesh.from_dp(size=0, devices=devices)
    with pytest.raises(ValueError, match="exceeds available device count"):
        ParallelMesh.from_dp(size=5, devices=devices)
    with pytest.raises(ValueError, match="axis_name must be non-empty"):
        ParallelMesh.from_dp(devices=devices, axis_name="")


def test_data_parallel_strategy() -> None:
    devices = _mock_devices(4)
    mesh = ParallelMesh.from_axes(devices=devices, data=2, fsdp=2)
    dp_strat = DataParallelStrategy(mesh=mesh, axis_name="data")

    assert dp_strat.batch_sharding() == NamedSharding(mesh.mesh, P("data"))
    assert dp_strat.batch_sharding(None) == NamedSharding(mesh.mesh, P("data", None))
    assert dp_strat.replicated == NamedSharding(mesh.mesh, P())

    with pytest.raises(ValueError, match="axis 'invalid' is not present"):
        DataParallelStrategy(mesh=mesh, axis_name="invalid")


def test_zero_sharding_strategy_contracts_and_specs() -> None:
    devices = _mock_devices(4)
    mesh = ParallelMesh.from_axes(devices=devices, data=2, fsdp=2)

    for stage in ZeroStage:
        zero_strat = ZeroShardingStrategy(mesh=mesh, stage=stage, axis_name="fsdp")
        contract = zero_strat.zero_contract()
        assert isinstance(contract, ZeroPartitionContract)
        assert contract.stage == stage
        assert contract.axis_name == "fsdp"

        param_sharding = zero_strat.zero_parameter_sharding()
        optim_sharding = zero_strat.zero_optimizer_sharding()
        grad_sharding = zero_strat.zero_gradient_sharding()

        if stage is ZeroStage.REPLICATED:
            assert param_sharding.spec == P()
            assert optim_sharding.spec == P()
            assert grad_sharding.spec == P()
        elif stage is ZeroStage.OPTIMIZER_STATE:
            assert param_sharding.spec == P()
            assert optim_sharding.spec == P("fsdp")
            assert grad_sharding.spec == P()
        elif stage is ZeroStage.GRADIENT:
            assert param_sharding.spec == P()
            assert optim_sharding.spec == P("fsdp")
            assert grad_sharding.spec == P("fsdp")
        elif stage is ZeroStage.PARAMETER:
            assert param_sharding.spec == P("fsdp")
            assert optim_sharding.spec == P("fsdp")
            assert grad_sharding.spec == P("fsdp")

    # State specs
    zero_rep = ZeroShardingStrategy(
        mesh=mesh, stage=ZeroStage.REPLICATED, axis_name="fsdp"
    )
    rep_specs = zero_rep.zero_state_specs(2, 4)
    assert rep_specs == (P(), P(), P(), P())

    zero_opt = ZeroShardingStrategy(
        mesh=mesh, stage=ZeroStage.OPTIMIZER_STATE, axis_name="fsdp"
    )
    opt_specs = zero_opt.zero_state_specs(2, 4)
    assert opt_specs == (P(), P(), P("fsdp"), P("fsdp"))

    zero_param = ZeroShardingStrategy(
        mesh=mesh, stage=ZeroStage.PARAMETER, axis_name="fsdp"
    )
    param_specs = zero_param.zero_state_specs(2, 4)
    assert param_specs == (P("fsdp"), P("fsdp"), P("fsdp"), P("fsdp"))

    # Grad specs
    assert zero_rep.zero_gradient_specs(2) == (P(), P())
    assert zero_opt.zero_gradient_specs(2) == (P(), P())
    zero_grad = ZeroShardingStrategy(
        mesh=mesh, stage=ZeroStage.GRADIENT, axis_name="fsdp"
    )
    assert zero_grad.zero_gradient_specs(2) == (P("fsdp"), P("fsdp"))
    assert zero_param.zero_gradient_specs(2) == (P("fsdp"), P("fsdp"))

    with pytest.raises(ValueError, match="invalid leaf counts"):
        zero_rep.zero_state_specs(5, 4)
    with pytest.raises(ValueError, match="leaf_count must be non-negative"):
        zero_rep.zero_gradient_specs(-1)


def test_zero_sharding_strategy_from_execution_plan() -> None:
    devices = _mock_devices(4)
    plan = TrainingExecutionPlan(
        zero=ZeroExecutionPolicy(ZeroStage.OPTIMIZER_STATE, axis_name="fsdp"),
        accumulation=GradientAccumulationPolicy(1),
        update=MacroUpdatePolicy(True),
    )
    mesh = ParallelMesh.from_axes(devices=devices, fsdp=4)
    strategy = ZeroShardingStrategy.from_execution_plan(plan, mesh)
    assert strategy.stage == ZeroStage.OPTIMIZER_STATE
    assert strategy.axis_name == "fsdp"
    assert strategy.mesh is mesh


def test_checkpoint_sharding_strategy() -> None:
    devices = _mock_devices(4)
    mesh = ParallelMesh.from_axes(devices=devices, data=2, fsdp=2)
    cp_strat = CheckpointShardingStrategy(mesh=mesh, axis_name="fsdp")

    # (8, 16): dim 0 is 8 % 2 == 0 -> P('fsdp')
    assert cp_strat.checkpoint_sharding((8, 16)).spec == P("fsdp")
    # (1, 16): dim 0 is 1 % 2 != 0, dim 1 is 16 % 2 == 0 -> P(None, 'fsdp')
    assert cp_strat.checkpoint_sharding((1, 16)).spec == P(None, "fsdp")
    # (3, 5): indivisible -> P()
    assert cp_strat.checkpoint_sharding((3, 5)).spec == P()
    # (): scalar -> P()
    assert cp_strat.checkpoint_sharding(()).spec == P()


def test_concrete_model_and_optimizer_sharding_with_strategy() -> None:
    real_devices = jax.devices()
    if len(real_devices) < 2:
        pytest.skip("Test requires at least 2 real devices")
    mesh = ParallelMesh.from_axes(devices=real_devices[:2], fsdp=2)
    zero_param = ZeroShardingStrategy(
        mesh=mesh, stage=ZeroStage.PARAMETER, axis_name="fsdp"
    )
    zero_opt = ZeroShardingStrategy(
        mesh=mesh, stage=ZeroStage.OPTIMIZER_STATE, axis_name="fsdp"
    )

    model = SimpleModel(rngs=nnx.Rngs(0))
    opt = create_nnx_optimizer(model, optax.adam(1e-3))

    sharded_model = zero_param.shard_model(model)
    assert isinstance(sharded_model, SimpleModel)
    assert sharded_model.w1.sharding.spec == P("fsdp")

    sharded_opt = zero_opt.shard_optimizer(opt)
    assert isinstance(sharded_opt, nnx.Optimizer)
    _, opt_state = nnx.split(sharded_opt)
    for _, leaf in nnx.to_flat_state(opt_state):
        if leaf.ndim == 0:
            assert leaf.sharding.spec == P()
        else:
            assert leaf.sharding.spec == P("fsdp")


def test_nnx_training_state_definition_checkpoint_layout_with_parallel_mesh() -> None:
    devices = _mock_devices(4)
    mesh = ParallelMesh.from_axes(devices=devices, data=2, fsdp=2)

    model_def = NnxModelDefinition.from_initializer(
        lambda: SimpleModel(rngs=nnx.Rngs(0))
    )
    state_def = NnxTrainingStateDefinition(
        model_def,
        lambda mod: create_nnx_optimizer(mod, optax.adam(1e-3)),
    )

    layout = state_def.checkpoint_layout(mesh, axis_name="fsdp")
    assert "step" in layout
    assert "model_state" in layout
    assert "optimizer_state" in layout

    assert layout["step"].sharding == mesh.replicated
    assert layout["step"].shape == ()

    for _, leaf in nnx.to_flat_state(layout["model_state"]):
        assert leaf.sharding.spec == P("fsdp")

    for _, leaf in nnx.to_flat_state(layout["optimizer_state"]):
        if len(leaf.shape) == 0:
            assert leaf.sharding.spec == P()
        else:
            assert leaf.sharding.spec == P("fsdp")

    sharded_abstract_model = state_def.shard_model(
        mesh, stage=ZeroStage.PARAMETER, axis_name="fsdp"
    )
    for _, leaf in nnx.to_flat_state(sharded_abstract_model):
        assert leaf.sharding.spec == P("fsdp")

    sharded_abstract_opt = state_def.shard_optimizer(
        mesh, stage=ZeroStage.OPTIMIZER_STATE, axis_name="fsdp"
    )
    for _, leaf in nnx.to_flat_state(sharded_abstract_opt):
        if len(leaf.shape) == 0:
            assert leaf.sharding.spec == P()
        else:
            assert leaf.sharding.spec == P("fsdp")


def test_require_distributed_execution_with_parallel_mesh() -> None:
    real_devices = jax.devices()
    if len(real_devices) < 2:
        pytest.skip("Test requires at least 2 real devices")
    mesh = ParallelMesh.from_dp(devices=real_devices[:2], axis_name="dp")

    # Must succeed without error when passing ParallelMesh
    require_distributed_execution(
        mesh=mesh, zero_stage=ZeroStage.REPLICATED, axis_name="dp"
    )

    with pytest.raises(ValueError, match="requires mesh axis 'dp' > 1"):
        single_device_mesh = ParallelMesh.from_dp(
            size=1, devices=real_devices[:2], axis_name="dp"
        )
        require_distributed_execution(
            mesh=single_device_mesh,
            zero_stage=ZeroStage.REPLICATED,
            axis_name="dp",
        )
