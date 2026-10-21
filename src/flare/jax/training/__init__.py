from __future__ import annotations

from flare.jax.layers.state import NnxModelDefinition
from flare.jax.mesh import state_with_sharding
from flare.jax.training.engines import (
    DistributedNnxTrainEngine,
    ReplicatedNnxTrainEngine,
    TrainStepResult,
)
from flare.jax.training.gradients import (
    accumulate_gradients,
    as_float_gradient,
    build_donated_gradient_accumulator,
    mean_accumulated_gradients,
    normalize_input_gradients,
    partition_input_gradients,
    validate_input_gradient_keys,
)
from flare.jax.training.nnx import (
    NnxAbstractState,
    NnxShardUpdatePlan,
    NnxTrainingStateDefinition,
)
from flare.jax.training.optimizers import (
    create_adagrad_optimizer,
    create_adam_optimizer,
    create_adamw_optimizer,
    create_default_optimizer,
    create_nnx_optimizer,
    create_rmsprop_optimizer,
    create_sgd_optimizer,
)
from flare.jax.training.precision import Fp32Param, PrecisionPolicy
from flare.jax.training.state import (
    DictPacker,
    FlatStateDtypeBucketLayout,
    FlatStateLayout,
    FlatStateLeafLayout,
    NnxStatePacker,
)
from flare.jax.training.steppers import (
    MacroUpdateOutput,
    MicroStepKernels,
    build_shard_map_macro_update,
    build_shard_map_micro_accumulate,
    build_single_step_dp_train_step,
    compute_loss_and_gradients,
    compute_module_and_input_gradients,
    require_distributed_execution,
)
from flare.jax.training.strategy import (
    CheckpointShardingStrategy,
    DataParallelStrategy,
    ShardingStrategy,
    ZeroShardingStrategy,
)

__all__ = [
    "CheckpointShardingStrategy",
    "DataParallelStrategy",
    "DictPacker",
    "DistributedNnxTrainEngine",
    "FlatStateDtypeBucketLayout",
    "FlatStateLayout",
    "FlatStateLeafLayout",
    "Fp32Param",
    "MacroUpdateOutput",
    "MicroStepKernels",
    "NnxAbstractState",
    "NnxModelDefinition",
    "NnxShardUpdatePlan",
    "NnxStatePacker",
    "NnxTrainingStateDefinition",
    "PrecisionPolicy",
    "ReplicatedNnxTrainEngine",
    "ShardingStrategy",
    "TrainStepResult",
    "ZeroShardingStrategy",
    "accumulate_gradients",
    "as_float_gradient",
    "build_donated_gradient_accumulator",
    "build_shard_map_macro_update",
    "build_shard_map_micro_accumulate",
    "build_single_step_dp_train_step",
    "compute_loss_and_gradients",
    "compute_module_and_input_gradients",
    "create_adagrad_optimizer",
    "create_adam_optimizer",
    "create_adamw_optimizer",
    "create_default_optimizer",
    "create_nnx_optimizer",
    "create_rmsprop_optimizer",
    "create_sgd_optimizer",
    "mean_accumulated_gradients",
    "normalize_input_gradients",
    "partition_input_gradients",
    "require_distributed_execution",
    "state_with_sharding",
    "validate_input_gradient_keys",
]
