from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from flax import nnx
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from flare.jax.mesh import ParallelMesh
from flare.jax.training.state.packers import NnxStatePacker
from flare.jax.training.zero import reduce_accumulated_gradients
from flare.training.zero.contracts import ZeroStage


@dataclass(frozen=True)
class MicroStepKernels:
    first: Callable[..., Any]
    accumulate: Callable[..., Any]
    diagnostic_first: Callable[..., Any] | None = None
    diagnostic_accumulate: Callable[..., Any] | None = None


class MacroUpdateOutput(NamedTuple):
    state: Any
    normalized_gradients: Any


def compute_loss_and_gradients(
    module: nnx.Module,
    forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
    loss_fn: Callable[[Any, Mapping[str, Any]], Any],
    inputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[tuple[Any, Any], tuple[Any, Mapping[str, Any]]]:
    """Compute loss, predictions, and gradients with respect to module parameters and inputs."""
    if not isinstance(module, nnx.Module):
        raise TypeError(f"module must be flax.nnx.Module, got {type(module).__name__}")
    if not callable(forward_fn):
        raise TypeError("forward_fn must be callable")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("inputs and targets must be mappings")

    def loss_with_aux(
        current_module: nnx.Module,
        differentiable_inputs: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        predictions = forward_fn(current_module, differentiable_inputs)
        loss = loss_fn(predictions, targets)
        return loss, predictions

    grad_fn = nnx.value_and_grad(
        loss_with_aux,
        argnums=(0, 1),
        has_aux=True,
        allow_int=True,
    )
    (loss, predictions), (param_grads, input_grads) = grad_fn(module, inputs)
    return (loss, predictions), (param_grads, input_grads)


def compute_module_and_input_gradients(
    module: nnx.Module,
    forward_fn: Callable[[nnx.Module, Mapping[str, Any]], Any],
    loss_fn: Callable[[Any, Any], Any],
    input_gradient_inputs: Mapping[str, Any],
    module_inputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[tuple[Any, Any], tuple[Any, Mapping[str, Any]]]:
    """Compute value and gradient for module parameters and selected inputs."""

    if not callable(forward_fn):
        raise TypeError("forward_fn must be callable")

    def merged_forward_fn(m: nnx.Module, selected_inputs: Mapping[str, Any]) -> Any:
        full_inputs = {**module_inputs, **selected_inputs}
        return forward_fn(m, full_inputs)

    return compute_loss_and_gradients(
        module=module,
        forward_fn=merged_forward_fn,
        loss_fn=loss_fn,
        inputs=input_gradient_inputs,
        targets=targets,
    )


def build_shard_map_micro_accumulate(
    micro_step_fn: Callable,
    mesh: Mesh | ParallelMesh,
    flat_state_in_spec: Any,
    model_state_out_spec: Any,
    gradient_out_spec: Any,
    input_specs: Any,
    input_gradient_specs: Any,
    target_specs: tuple[Any, ...],
    *,
    reduce_gradients_fn: Callable[[Any], Any],
    transform_raw_gradients_fn: Callable[[Any], Any] | None = None,
    axis_name: str = "dp",
    retain_current_gradients: bool = False,
) -> MicroStepKernels:
    """Build the first and donated accumulating distributed micro kernels."""
    mesh = mesh.mesh if isinstance(mesh, ParallelMesh) else mesh
    require_distributed_execution(
        mesh=mesh, zero_stage=ZeroStage.REPLICATED, axis_name=axis_name
    )
    if not callable(micro_step_fn):
        raise TypeError("micro_step_fn must be callable")
    if not callable(reduce_gradients_fn):
        raise TypeError("reduce_gradients_fn must be callable")
    if retain_current_gradients and not callable(transform_raw_gradients_fn):
        raise TypeError("diagnostic micro kernels require transform_raw_gradients_fn")
    if not isinstance(input_gradient_specs, tuple):
        raise TypeError("input_gradient_specs must be a tuple")
    if not isinstance(target_specs, tuple):
        raise TypeError("target_specs must be a tuple")

    def _reduce(current_gradients: Any) -> Any:
        return reduce_gradients_fn(current_gradients)

    def _first(*args: Any):
        updated_model, current_gradients, input_gradients, loss, predictions = (
            micro_step_fn(*args)
        )
        return (
            updated_model,
            _reduce(current_gradients),
            input_gradients,
            loss,
            predictions,
        )

    def _accumulate(*args: Any):
        *micro_args, accumulated_gradients = args
        updated_model, current_gradients, input_gradients, loss, predictions = (
            micro_step_fn(*micro_args)
        )
        reduced_gradients = _reduce(current_gradients)
        accumulated_gradients = jax.tree_util.tree_map(
            lambda total, current: total + current,
            accumulated_gradients,
            reduced_gradients,
        )
        return (
            updated_model,
            accumulated_gradients,
            input_gradients,
            loss,
            predictions,
        )

    first_smap = shard_map(
        _first,
        mesh=mesh,
        in_specs=(flat_state_in_spec, input_specs, target_specs),
        out_specs=(
            model_state_out_spec,
            gradient_out_spec,
            input_gradient_specs,
            P(),
            P(axis_name, None),
        ),
        check_vma=False,
    )
    accumulate_smap = shard_map(
        _accumulate,
        mesh=mesh,
        in_specs=(
            flat_state_in_spec,
            input_specs,
            target_specs,
            gradient_out_spec,
        ),
        out_specs=(
            model_state_out_spec,
            gradient_out_spec,
            input_gradient_specs,
            P(),
            P(axis_name, None),
        ),
        check_vma=False,
    )
    first_jit = jax.jit(first_smap)
    accumulate_jit = jax.jit(accumulate_smap, donate_argnums=(4,))
    if not retain_current_gradients:
        return MicroStepKernels(first=first_jit, accumulate=accumulate_jit)
    if transform_raw_gradients_fn is None:
        raise TypeError("diagnostic micro kernels require transform_raw_gradients_fn")

    def _first_with_current(*args: Any):
        updated_model, current_gradients, input_gradients, loss, predictions = (
            micro_step_fn(*args)
        )
        reduced_gradients = _reduce(current_gradients)
        return (
            updated_model,
            reduced_gradients,
            transform_raw_gradients_fn(current_gradients),
            reduced_gradients,
            input_gradients,
            loss,
            predictions,
        )

    def _accumulate_with_current(*args: Any):
        *micro_args, accumulated_gradients = args
        updated_model, current_gradients, input_gradients, loss, predictions = (
            micro_step_fn(*micro_args)
        )
        reduced_gradients = _reduce(current_gradients)
        new_accumulated_gradients = jax.tree_util.tree_map(
            lambda total, current: total + current,
            accumulated_gradients,
            reduced_gradients,
        )
        return (
            updated_model,
            new_accumulated_gradients,
            transform_raw_gradients_fn(current_gradients),
            reduced_gradients,
            input_gradients,
            loss,
            predictions,
        )

    accumulate_with_current = jax.jit(
        shard_map(
            _accumulate_with_current,
            mesh=mesh,
            in_specs=(
                flat_state_in_spec,
                input_specs,
                target_specs,
                gradient_out_spec,
            ),
            out_specs=(
                model_state_out_spec,
                gradient_out_spec,
                tuple(P(axis_name, None) for _ in gradient_out_spec),
                gradient_out_spec,
                input_gradient_specs,
                P(),
                P(axis_name, None),
            ),
            check_vma=False,
        ),
        donate_argnums=(4,),
    )
    return MicroStepKernels(
        first=first_jit,
        accumulate=accumulate_jit,
        diagnostic_first=jax.jit(
            shard_map(
                _first_with_current,
                mesh=mesh,
                in_specs=(flat_state_in_spec, input_specs, target_specs),
                out_specs=(
                    model_state_out_spec,
                    gradient_out_spec,
                    tuple(P(axis_name, None) for _ in gradient_out_spec),
                    gradient_out_spec,
                    input_gradient_specs,
                    P(),
                    P(axis_name, None),
                ),
                check_vma=False,
            )
        ),
        diagnostic_accumulate=accumulate_with_current,
    )


def build_shard_map_macro_update(
    optimizer_update_fn: Callable,
    mesh: Mesh | ParallelMesh,
    flat_state_in_spec: Any,
    acc_grads_state_spec: Any,
    zero_stage: ZeroStage | None = None,
    axis_name: str = "dp",
    state_in_shardings: Any | None = None,
    gradient_in_shardings: Any | None = None,
    accumulation_steps_sharding: Any | None = None,
    state_out_shardings: Any | None = None,
    return_normalized_gradients: bool = False,
    requires_axis_context: bool = False,
) -> Callable:
    """Build shard_map macro update function that synchronizes accumulated gradients according to ZeRO stage."""
    if zero_stage is not None:
        zero_stage = ZeroStage(zero_stage)
    mesh = mesh.mesh if isinstance(mesh, ParallelMesh) else mesh
    require_distributed_execution(
        mesh=mesh,
        zero_stage=ZeroStage.REPLICATED if zero_stage is None else zero_stage,
        axis_name=axis_name,
    )

    def _post_process_state(state):
        if zero_stage == ZeroStage.PARAMETER:
            return jax.tree_util.tree_map(
                lambda p: jax.lax.all_gather(p, axis_name, axis=0, tiled=True),
                state,
            )
        return state

    def _macro_update_fn(flat_state_in, acc_grads_state, accumulation_steps):
        if zero_stage is not None:
            acc_grads_state = reduce_accumulated_gradients(
                acc_grads_state, stage=zero_stage, axis_name=axis_name
            )

        update_result = optimizer_update_fn(
            flat_state_in, acc_grads_state, accumulation_steps
        )
        if return_normalized_gradients:
            new_flat_state, normalized_gradients = update_result
            return MacroUpdateOutput(
                _post_process_state(new_flat_state), normalized_gradients
            )
        return _post_process_state(update_result)

    if (
        state_in_shardings is None
        and gradient_in_shardings is None
        and accumulation_steps_sharding is None
        and state_out_shardings is None
    ):
        macro_smap = shard_map(
            _macro_update_fn,
            mesh=mesh,
            in_specs=(flat_state_in_spec, acc_grads_state_spec, P()),
            out_specs=MacroUpdateOutput(flat_state_in_spec, acc_grads_state_spec)
            if return_normalized_gradients
            else flat_state_in_spec,
            check_vma=False,
        )

        donate_argnums = (0, 1) if return_normalized_gradients else (0,)

        @jax.jit(donate_argnums=donate_argnums)
        def macro_update(flat_state_in, acc_grads_state, accumulation_steps):
            return macro_smap(flat_state_in, acc_grads_state, accumulation_steps)

        return macro_update

    if (
        state_in_shardings is None
        or gradient_in_shardings is None
        or accumulation_steps_sharding is None
        or state_out_shardings is None
    ):
        raise ValueError(
            "all macro state and scalar shardings must be provided together"
        )

    output_shardings = MacroUpdateOutput(state_out_shardings, gradient_in_shardings)
    if not return_normalized_gradients:
        output_shardings = state_out_shardings
    donate_argnums = (0, 1) if return_normalized_gradients else (0,)
    if requires_axis_context:
        macro_smap = shard_map(
            _macro_update_fn,
            mesh=mesh,
            in_specs=(
                tuple(sharding.spec for sharding in state_in_shardings),
                tuple(sharding.spec for sharding in gradient_in_shardings),
                accumulation_steps_sharding.spec,
            ),
            out_specs=(
                MacroUpdateOutput(
                    tuple(sharding.spec for sharding in state_out_shardings),
                    tuple(sharding.spec for sharding in gradient_in_shardings),
                )
                if return_normalized_gradients
                else tuple(sharding.spec for sharding in state_out_shardings)
            ),
            check_vma=False,
        )

        @jax.jit(donate_argnums=donate_argnums)
        def macro_update_with_axis_context(
            flat_state_in, acc_grads_state, accumulation_steps
        ):
            return macro_smap(flat_state_in, acc_grads_state, accumulation_steps)

        return macro_update_with_axis_context

    macro_update = jax.jit(
        _macro_update_fn,
        in_shardings=(
            state_in_shardings,
            gradient_in_shardings,
            accumulation_steps_sharding,
        ),
        out_shardings=output_shardings,
        donate_argnums=donate_argnums,
    )

    return macro_update


def build_single_step_dp_train_step(
    *,
    packer: NnxStatePacker,
    mesh: Mesh,
    loss_fn: Callable[[nnx.Module, Any, Any], tuple[Any, Any]],
    input_specs: Any,
    target_specs: Any,
    prediction_specs: Any,
    module_group: str = "module",
    optimizer_group: str = "optimizer",
    axis_name: str = "dp",
) -> Callable:
    """Build a single-step data-parallel (DP) NNX training step without gradient accumulation."""
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be a jax.sharding.Mesh, got {type(mesh).__name__}")
    mesh_size = int(mesh.shape.get(axis_name, 0))
    if mesh_size <= 1:
        raise ValueError(f"mesh axis {axis_name!r} must have size greater than one")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    module_leaf_count = packer.leaf_count(module_group)
    optimizer_leaf_count = packer.leaf_count(optimizer_group)
    flat_state_specs = tuple(
        P() for _ in range(module_leaf_count + optimizer_leaf_count)
    )

    def _step(flat_state: tuple[Any, ...], inputs: Any, targets: Any):
        objects = packer.unflatten(flat_state)
        module = objects[module_group]
        optimizer = objects[optimizer_group]

        def differentiated(current_module: nnx.Module):
            return loss_fn(current_module, inputs, targets)

        (loss, predictions), gradients = nnx.value_and_grad(
            differentiated,
            has_aux=True,
            allow_int=True,
        )(module)
        gradients = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name), gradients
        )
        optimizer.update(module, gradients)
        return (
            packer.repack(**{module_group: module, optimizer_group: optimizer}),
            loss,
            predictions,
        )

    mapped = shard_map(
        _step,
        mesh=mesh,
        in_specs=(flat_state_specs, input_specs, target_specs),
        out_specs=(flat_state_specs, P(), prediction_specs),
        check_vma=False,
    )

    @jax.jit(donate_argnums=(0,))
    def step(flat_state: tuple[Any, ...], inputs: Any, targets: Any):
        return mapped(flat_state, inputs, targets)

    return step


def require_distributed_execution(
    *, mesh: Mesh | ParallelMesh, zero_stage: ZeroStage, axis_name: str = "dp"
) -> None:
    """Validate that a caller is using the explicit collective execution seam."""
    raw_mesh = mesh.mesh if isinstance(mesh, ParallelMesh) else mesh
    if not isinstance(raw_mesh, Mesh):
        raise TypeError(
            f"mesh must be a jax.sharding.Mesh or ParallelMesh, got {type(mesh).__name__}"
        )
    if not isinstance(zero_stage, ZeroStage):
        raise TypeError(
            f"zero_stage must be ZeroStage, got {type(zero_stage).__name__}"
        )
    size = int(raw_mesh.shape.get(axis_name, 0))
    if size <= 1:
        raise ValueError(f"distributed execution requires mesh axis {axis_name!r} > 1")


__all__ = [
    "MacroUpdateOutput",
    "MicroStepKernels",
    "build_shard_map_macro_update",
    "build_shard_map_micro_accumulate",
    "build_single_step_dp_train_step",
    "compute_loss_and_gradients",
    "compute_module_and_input_gradients",
    "require_distributed_execution",
]
