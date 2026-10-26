from __future__ import annotations

from typing import Any

import numpy as np
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointSaveMode,
    PartitionSpec,
)

from flare.runtime.fluxway.checkpoint.finalize import (
    checkpoint_finalizer_registration,
)
from flare.runtime.fluxway.checkpoint.sparse import (
    sparse_checkpoint_registration,
)
from flare.runtime.fluxway.workers.base import (
    assigned_worker_cpu_count,
    require_operator_step,
)
from flare.runtime.sparse_precision import (
    prepare_sparse_native_gradient,
    require_finite_sparse_values,
    require_key_dtype,
    require_sparse_gradient_dtype,
    require_sparse_gradient_transport_dtype,
    require_sparse_key_dtype,
    require_sparse_optimizer_dtype,
    require_sparse_value_dtype,
)
from flare.training.components.sparse_table.contracts import SparseOptimizerConfig
from flare.training.execution import OptimizerUpdateWindow
from fluxway.common.bind import bind_spmd
from fluxway.common.decorators import requires_worker_process
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy, StatePolicy
from fluxway.common.resource_registry import get_active_resource_registry
from fluxway.common.tensor_backend import get_backend, prepare_dlpack_input
from fluxway.common.types.tensor import TransportTensor


def register_sparse_partition_resource(
    resource_id: str,
    assignment_spec: PartitionSpec,
) -> None:
    """Create the owner strategy once per Worker execution context."""
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("sparse partition resource_id must be non-empty")
    if not isinstance(assignment_spec, PartitionSpec):
        raise TypeError("assignment_spec must be PartitionSpec")
    get_active_resource_registry().register(
        "sparse_embedding", resource_id, se.create_strategy(assignment_spec)
    )


def sparse_partition_resource_strategy(resource_id: str) -> dict[str, Any]:
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("sparse partition resource_id must be non-empty")
    return {
        "type": "function",
        "target": {
            "namespace": "sparse_embedding",
            "resource_id": resource_id,
        },
    }


def create_sparse_optimizer(config: SparseOptimizerConfig) -> se.Optimizer:
    policy_factories = {
        "constant": se.ConstantLrPolicy,
        "sqrt_decay": se.SqrtDecayLrPolicy,
        "log_decay": se.LogDecayLrPolicy,
        "inverse_decay": se.InverseDecayLrPolicy,
        "cosine_frequency": lambda: se.CosineFrequencyLrPolicy(
            min_lr_ratio=config.cosine_min_lr_ratio,
            max_score=config.cosine_max_score,  # pyrefly: ignore [bad-argument-type]
        ),
        "custom_lut": lambda: se.CustomLutLrPolicy(lut=config.custom_lut),
    }
    try:
        dynamic_lr_policy = policy_factories[config.dynamic_lr_policy]()
    except KeyError as error:
        raise ValueError(
            f"unsupported sparse dynamic learning-rate policy "
            f"{config.dynamic_lr_policy!r}"
        ) from error
    optimizer_factories = {
        "adam": lambda: se.Adam(
            lr=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            dynamic_lr_policy=dynamic_lr_policy,
        ),
        "sgd": lambda: se.Sgd(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            dynamic_lr_policy=dynamic_lr_policy,
        ),
    }
    try:
        return optimizer_factories[config.kind]()
    except KeyError as error:
        raise ValueError(f"unsupported sparse optimizer {config.kind!r}") from error


def sparse_lookup_async_policy(max_staleness_steps: int) -> AsyncPolicy:
    if max_staleness_steps < 0:
        raise ValueError(
            f"max_staleness_steps must be non-negative, got {max_staleness_steps}"
        )
    return AsyncPolicy.bounded_stale(
        depth=max_staleness_steps + 1,
        max_staleness=max_staleness_steps,
        commit="relaxed",
        schedule=SchedulePolicy.deadline_boost(),
        state=StatePolicy.thread_safe(),
    )


def sparse_update_async_policy() -> AsyncPolicy:
    return AsyncPolicy.bounded_stale(
        depth=1,
        max_staleness=0,
        commit="relaxed",
        schedule=SchedulePolicy.deadline_boost(),
        state=StatePolicy.thread_safe(),
    )


@requires_worker_process
class DynamicSparseEmbeddingSPMD:
    _factory_key_dtype: str | None = None
    _factory_value_dtype: str | None = None
    _factory_gradient_dtype: str | None = None
    _factory_gradient_transport_dtype: str | None = None
    _factory_optimizer_dtype: str | None = None
    _factory_assignment_spec: PartitionSpec | None = None
    _factory_resource_id: str | None = None
    rank: int
    world_size: int
    ctx: Any

    def __init__(
        self,
        gname: str,
        spec: dict[str, Any],
        LookupOutput: type,
        execution_tracer: Any | None = None,
        validate_tensor_values: bool = False,
    ):
        self.gname = gname
        if not isinstance(validate_tensor_values, bool):
            raise TypeError("validate_tensor_values must be bool")
        if execution_tracer is not None:
            self.execution_tracer = execution_tracer
            prepare_trace = getattr(execution_tracer, "prepare", None)
            if callable(prepare_trace):
                prepare_trace()
        if validate_tensor_values:
            self.validate_tensor_values = True
        assignment_spec = type(self)._factory_assignment_spec
        resource_id = type(self)._factory_resource_id
        if not isinstance(assignment_spec, PartitionSpec):
            raise TypeError(
                f"SparseTable[{gname}] requires an owner PartitionSpec at worker bootstrap"
            )
        if not isinstance(resource_id, str) or not resource_id:
            raise TypeError(
                f"SparseTable[{gname}] requires a non-empty routing resource id"
            )
        register_sparse_partition_resource(resource_id, assignment_spec)
        self.embed_dims = spec["emb_dims"]
        self.value_dtype = require_sparse_value_dtype(spec["value_dtype"])
        self.gradient_dtype = require_sparse_gradient_dtype(spec["gradient_dtype"])
        self.gradient_transport_dtype = require_sparse_gradient_transport_dtype(
            spec["gradient_transport_dtype"]
        )
        factory_value_dtype = type(self)._factory_value_dtype
        if factory_value_dtype is not None and self.value_dtype != factory_value_dtype:
            raise ValueError(
                f"SparseTable[{self.gname}] value_dtype differs between factory "
                f"and spec: factory={factory_value_dtype!r} spec={self.value_dtype!r}"
            )
        factory_gradient_dtype = type(self)._factory_gradient_dtype
        if (
            factory_gradient_dtype is not None
            and self.gradient_dtype != factory_gradient_dtype
        ):
            raise ValueError(
                f"SparseTable[{self.gname}] gradient_dtype differs between factory "
                f"and spec: factory={factory_gradient_dtype!r} "
                f"spec={self.gradient_dtype!r}"
            )
        factory_gradient_transport_dtype = type(self)._factory_gradient_transport_dtype
        if (
            factory_gradient_transport_dtype is not None
            and self.gradient_transport_dtype != factory_gradient_transport_dtype
        ):
            raise ValueError(
                f"SparseTable[{self.gname}] gradient_transport_dtype differs between "
                f"factory and spec: factory={factory_gradient_transport_dtype!r} "
                f"spec={self.gradient_transport_dtype!r}"
            )
        self.optimizer_dtype = require_sparse_optimizer_dtype(spec["optimizer_dtype"])
        factory_optimizer_dtype = type(self)._factory_optimizer_dtype
        if (
            factory_optimizer_dtype is not None
            and self.optimizer_dtype != factory_optimizer_dtype
        ):
            raise ValueError(
                f"SparseTable[{self.gname}] optimizer_dtype differs between factory "
                f"and spec: factory={factory_optimizer_dtype!r} "
                f"spec={self.optimizer_dtype!r}"
            )
        self.LookupOutput = LookupOutput
        self.grad_accum_steps = spec["grad_accum_steps"]
        self.key_dtype = require_key_dtype(spec["key_dtype"])
        factory_key_dtype = type(self)._factory_key_dtype
        if factory_key_dtype is not None and self.key_dtype != factory_key_dtype:
            raise ValueError(
                f"SparseTable[{self.gname}] key_dtype differs between "
                f"factory and spec: factory={factory_key_dtype!r} "
                f"spec={self.key_dtype!r}"
            )

        import ray

        assigned_cpus = assigned_worker_cpu_count(
            ray.get_runtime_context(), role_name="SparseTable worker"
        )

        optimizer_enabled = spec.get("optimizer_enabled")
        if not isinstance(optimizer_enabled, bool):
            raise TypeError(
                f"[SparseTable.__init__:{self.gname}] optimizer_enabled must be bool"
            )
        sparse_optimizer = spec.get("sparse_optimizer")
        if not optimizer_enabled and sparse_optimizer is not None:
            raise ValueError(
                f"[SparseTable.__init__:{self.gname}] sparse_optimizer must be omitted "
                "when optimizer_enabled=False"
            )
        embedding_config = se.EmbeddingConfig(
            dims=self.embed_dims,
            thread_budget=assigned_cpus,
            value_dtype=self.value_dtype,
            gradient_dtype=self.gradient_dtype,
            optimizer_dtype=self.optimizer_dtype,
            key_dtype=self.key_dtype,
            storage=se.AutoStorage(),
            enable_optimizer=optimizer_enabled,
        )
        store_kwargs: dict[str, Any] = {
            "embedding_config": embedding_config,
            "initializer_config": se.Uniform(min_val=-0.01, max_val=0.01),
            "score_policy": "lfu",
        }
        if optimizer_enabled:
            if not isinstance(sparse_optimizer, SparseOptimizerConfig):
                raise TypeError(
                    f"[SparseTable.__init__:{self.gname}] sparse_optimizer must be "
                    f"SparseOptimizerConfig when optimizer_enabled=True, got "
                    f"{type(sparse_optimizer)!r}."
                )
            store_kwargs["optimizer_config"] = create_sparse_optimizer(sparse_optimizer)
        self.store = se.EmbeddingStore(
            **store_kwargs,
        )
        if optimizer_enabled:
            required_update_methods = (
                "lookup_or_insert",
                "accumulate_gradients",
                "optimizer_step",
            )
            missing_update_methods = tuple(
                name
                for name in required_update_methods
                if not hasattr(self.store, name)
            )
            if missing_update_methods:
                raise RuntimeError(
                    "sparse embedding update API is incompatible: "
                    f"missing={missing_update_methods!r}; "
                    f"required={required_update_methods!r}"
                )
        sparse_reserve_capacity = int(spec["sparse_reserve_capacity"])
        if sparse_reserve_capacity < 0:
            raise ValueError(
                f"[SparseTable.__init__:{self.gname}] sparse_reserve_capacity must be "
                f"non-negative, got {sparse_reserve_capacity}."
            )
        if sparse_reserve_capacity > 0:
            for dim in self.embed_dims:
                self.store.reserve(sparse_reserve_capacity, dim)

        weight_source = spec.get("weight_source")
        if weight_source is not None:
            if not hasattr(weight_source, "sparse"):
                raise TypeError(
                    "sparse weight source must expose a sparse group mapping"
                )
            try:
                sparse_adapter = weight_source.sparse[self.gname]
            except KeyError as error:
                raise KeyError(
                    f"weight source has no sparse adapter for {self.gname!r}"
                ) from error
            restore_payload = spec.get("sparse_restore_payload")
            if not isinstance(restore_payload, bytes) or not restore_payload:
                raise TypeError(
                    "sparse worker requires a preselected opaque restore payload"
                )
            sparse_adapter.load(self.store, restore_payload)

        self._optimizer_enabled = optimizer_enabled
        self._forward_keys_cache: dict[int, dict[int, np.ndarray]] = {}
        self._current_step: int | None = None
        self._last_committed_step: int | None = None
        self._active_optimizer_step: int | None = None
        self._steps_since_commit = 0
        self._committed_steps = 0
        if execution_tracer is not None and execution_tracer.include_values:
            self._traced_gradients: dict[int, dict[int, np.ndarray]] = {}

    async def _read_embeddings_for_audit(
        self, keys: np.ndarray, dim: int
    ) -> np.ndarray:
        expected_dtype = np.dtype(self.value_dtype)
        if keys.ndim != 1 or not np.issubdtype(keys.dtype, np.integer):
            raise TypeError(
                f"[SparseTable.audit:{self.gname}] keys must be a 1-D integer array, "
                f"got shape={keys.shape} dtype={keys.dtype}"
            )
        if keys.size == 0:
            return np.empty((0, dim), dtype=expected_dtype)
        values_capsule, _ = await self.store.async_lookup(
            prepare_dlpack_input(keys), dim
        )
        backend = get_backend()
        values = backend.from_dlpack(values_capsule).reshape((keys.size, dim))
        embeddings = np.asarray(backend.to_numpy(values))
        if np.dtype(embeddings.dtype) != expected_dtype:
            raise TypeError(
                f"[SparseTable.audit:{self.gname}] lookup dtype mismatch: "
                f"expected={expected_dtype} actual={embeddings.dtype}"
            )
        if embeddings.shape != (keys.size, dim):
            raise RuntimeError(
                f"[SparseTable.audit:{self.gname}] lookup returned shape "
                f"{embeddings.shape}, expected {(keys.size, dim)}"
            )
        return embeddings

    async def lookup(self, **unique_ids: TransportTensor) -> Any:
        backend = get_backend()
        step = require_operator_step(
            self,
            f"SparseTable[{self.gname}].lookup",
        )
        if self._current_step is None or step != self._current_step:
            self._current_step = step

        results: dict[str, TransportTensor] = {}

        for dim in self.embed_dims:
            field_name = f"unique_ids_dim{dim}"
            try:
                tensor = unique_ids[field_name]
            except KeyError as error:
                raise ValueError(
                    f"SparseTable.lookup missing required field {field_name!r}"
                ) from error
            if tensor is None:
                raise ValueError(
                    f"SparseTable.lookup received None tensor for dim={dim}"
                )
            if not isinstance(tensor, TransportTensor):
                raise TypeError(
                    f"SparseTable.lookup received unsupported tensor type: {type(tensor)}"
                )

            keys = require_sparse_key_dtype(
                tensor.flat_values,
                self.key_dtype,
                boundary=f"SparseTable[{self.gname}].lookup",
            ).reshape(-1)
            self._forward_keys_cache.setdefault(step, {})[dim] = keys
            dlpack_capsule = prepare_dlpack_input(keys)
            embed_capsule, _ = await self.store.async_lookup_or_insert(
                dlpack_capsule, dim
            )
            embed_tensor = backend.from_dlpack(embed_capsule).reshape((-1, dim))

            tracer = getattr(self, "execution_tracer", None)
            if tracer is not None:
                tracer.sparse_lookup(
                    group=f"{self.gname}_dim_{dim}",
                    step=step,
                    spmd_id=f"Embedding_{self.gname}",
                    rank=self.rank,
                    keys=keys,
                    embeddings=embed_tensor,
                    input_tensor=tensor,
                )
            if getattr(self, "validate_tensor_values", False):
                embed_tensor = require_finite_sparse_values(
                    embed_tensor, self.value_dtype, boundary="sparse_lookup"
                )
            if np.dtype(embed_tensor.dtype) != np.dtype(self.value_dtype):
                raise TypeError(
                    "native sparse lookup returned an unexpected value dtype"
                )

            output_tensor = TransportTensor(
                flat_values=embed_tensor,
                origin_indices=tensor.origin_indices,
                routing_indices=tensor.routing_indices,
                global_shape=(
                    (tensor.global_shape[0], dim)
                    if tensor.global_shape is not None
                    else None
                ),
                receipt=tensor.receipt,
                routing_axes=tensor.routing_axes,
                source_step=tensor.source_step,
            )
            output_name = f"sparse_features_{self.gname}_dim_{dim}"
            results[output_name] = output_tensor

        return self.LookupOutput(**results)

    def _resolve_optimizer_window(self, window: OptimizerUpdateWindow) -> None:
        if window.accumulation_steps != self.grad_accum_steps:
            raise ValueError(
                f"SparseTable[{self.gname}].update accumulation mismatch: "
                f"window={window.accumulation_steps}, table={self.grad_accum_steps}"
            )
        expected_micro_step = self._steps_since_commit + 1
        if window.micro_step != expected_micro_step:
            raise ValueError(
                f"SparseTable[{self.gname}].update micro-step mismatch: "
                f"window={window.micro_step}, expected={expected_micro_step}"
            )
        expected_boundary = expected_micro_step == self.grad_accum_steps
        if window.at_boundary != expected_boundary:
            raise ValueError(
                f"SparseTable[{self.gname}].update boundary mismatch: "
                f"window={window.at_boundary}, expected={expected_boundary}"
            )
        if window.should_apply_update != expected_boundary:
            raise ValueError(
                f"SparseTable[{self.gname}].update apply flag mismatch: "
                f"window={window.should_apply_update}, expected={expected_boundary}"
            )
        if self._active_optimizer_step is None:
            self._active_optimizer_step = window.optimizer_step
        elif window.optimizer_step != self._active_optimizer_step:
            if self._steps_since_commit != 0:
                raise ValueError(
                    f"SparseTable[{self.gname}].update optimizer window changed "
                    "before sparse commit: "
                    f"window={window.optimizer_step}, active={self._active_optimizer_step}"
                )
            expected_optimizer_step = self._active_optimizer_step + 1
            if window.optimizer_step != expected_optimizer_step:
                raise ValueError(
                    f"SparseTable[{self.gname}].update optimizer window sequence "
                    f"mismatch: window={window.optimizer_step}, "
                    f"expected={expected_optimizer_step}"
                )
            self._active_optimizer_step = window.optimizer_step

    async def _accumulate_gradients(
        self,
        step: int,
        grads_by_dim: dict[int, list[tuple[np.ndarray, TransportTensor]]],
    ) -> None:
        traced_gradients = getattr(self, "_traced_gradients", None)
        backend = get_backend() if traced_gradients is not None else None
        if traced_gradients is not None:
            if backend is None:
                raise RuntimeError("sparse observer gradients require a tensor backend")
            for dim, grad_list in grads_by_dim.items():
                gradients_by_key = traced_gradients.setdefault(dim, {})
                for keys, tensor in grad_list:
                    if keys.size == 0:
                        continue
                    gradients_host = np.asarray(
                        backend.to_numpy(tensor.flat_values), dtype=np.float32
                    ).reshape(keys.size, dim)
                    for key, gradient in zip(keys, gradients_host, strict=True):
                        key_value = int(key)
                        if key_value in gradients_by_key:
                            gradients_by_key[key_value] += gradient
                        else:
                            gradients_by_key[key_value] = gradient.copy()
        for grad_list in grads_by_dim.values():
            for keys, tensor in grad_list:
                if keys.size > 0:
                    native_gradient = prepare_sparse_native_gradient(
                        tensor.flat_values,
                        transport_dtype=self.gradient_transport_dtype,
                        native_dtype=self.gradient_dtype,
                        validate_values=getattr(self, "validate_tensor_values", False),
                    )
                    grad_data = prepare_dlpack_input(native_gradient)
                    # The native store accumulates keyed sums. Replica or
                    # token normalization belongs to an explicit upstream
                    # contract; this path must not infer it from key counts.
                    await self.store.async_accumulate_gradients(keys, grad_data)

    async def _commit_optimizer_step(self, step: int, normalization_steps: int) -> None:
        if (
            isinstance(normalization_steps, bool)
            or not isinstance(normalization_steps, int)
            or normalization_steps <= 0
        ):
            raise ValueError("normalization_steps must be a positive integer")
        # This is the actual number of microsteps in this committed window,
        # including a 1/2-step tail. It is neither DP world size nor ID count.
        await self.store.async_optimizer_step(normalization_steps)
        self._last_committed_step = step
        self._steps_since_commit = 0
        self._committed_steps += 1

    async def update(self, **inputs: Any) -> None:
        if not self._optimizer_enabled:
            raise RuntimeError(
                f"SparseTable[{self.gname}].update requires optimizer_enabled=True"
            )
        operator_step = require_operator_step(
            self,
            f"SparseTable[{self.gname}].update",
        )
        expected_names = tuple(f"grads_dim{dim}" for dim in self.embed_dims) + (
            "optimizer_update_window",
        )
        actual_names = tuple(inputs)
        if set(actual_names) != set(expected_names):
            missing = tuple(name for name in expected_names if name not in inputs)
            unexpected = tuple(name for name in inputs if name not in expected_names)
            raise ValueError(
                f"SparseTable[{self.gname}].update requires named inputs "
                f"{expected_names}, actual={actual_names}, "
                f"missing={missing}, unexpected={unexpected}"
            )
        window_value = inputs["optimizer_update_window"]
        if isinstance(window_value, OptimizerUpdateWindow):
            window = window_value
        else:
            if isinstance(window_value, TransportTensor):
                payload = window_value.flat_values
            elif isinstance(window_value, (np.ndarray, list, tuple)):
                payload = window_value
            else:
                raise TypeError(
                    "SparseTable.update optimizer_update_window must be an "
                    "OptimizerUpdateWindow, TransportTensor, or integer sequence"
                )
            window = OptimizerUpdateWindow.decode(
                np.asarray(payload, dtype=np.int64).reshape(-1)
            )
        self._resolve_optimizer_window(window)

        raw_grads = tuple(inputs[f"grads_dim{dim}"] for dim in self.embed_dims)
        expected_grad_count = len(self.embed_dims)
        if len(raw_grads) != expected_grad_count:
            raise ValueError(
                f"SparseTable[{self.gname}].update expected {expected_grad_count} "
                f"gradients, got {len(raw_grads)}"
            )

        grads_to_accumulate: dict[int, list[tuple[np.ndarray, TransportTensor]]] = {
            dim: [] for dim in self.embed_dims
        }
        for dim, tensor in zip(self.embed_dims, raw_grads, strict=False):
            if tensor is None:
                raise ValueError(
                    f"SparseTable.update received None gradient tensor for dim={dim}"
                )
            if not isinstance(tensor, TransportTensor):
                raise TypeError(
                    f"SparseTable.update received unsupported gradient type: {type(tensor)}"
                )
            source_step = tensor.source_step
            if source_step is None:
                raise ValueError(
                    f"SparseTable.update gradient tensor missing source_step for dim={dim}"
                )
            keys_by_dim = self._forward_keys_cache.get(source_step)
            if keys_by_dim is None or dim not in keys_by_dim:
                raise ValueError(
                    f"SparseTable.update missing forward keys for dim={dim}, "
                    f"source_step={source_step}"
                )
            keys = keys_by_dim.pop(dim)
            if not keys_by_dim:
                del self._forward_keys_cache[source_step]
            expected_shape = (keys.size, dim)
            actual_shape = tuple(tensor.flat_values.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"SparseTable.update gradient shape {actual_shape} does not match "
                    f"forward keys {expected_shape} for dim={dim}"
                )
            grads_to_accumulate[dim].append((keys, tensor))
        await self._accumulate_gradients(operator_step, grads_to_accumulate)
        self._steps_since_commit += 1
        optimizer_applied = self._steps_since_commit >= self.grad_accum_steps
        if optimizer_applied:
            await self._commit_optimizer_step(
                window.optimizer_step, self._steps_since_commit
            )
        tracer = getattr(self, "execution_tracer", None)
        if tracer is not None:
            for dim, grad_list in grads_to_accumulate.items():
                for keys, tensor in grad_list:
                    embeddings_after = None
                    if tracer.include_values and keys.size > 0:
                        embeddings_after = await self._read_embeddings_for_audit(
                            keys, dim
                        )
                    tracer.sparse_update(
                        group=f"{self.gname}_dim_{dim}",
                        step=operator_step,
                        spmd_id=f"Embedding_{self.gname}",
                        rank=self.rank,
                        keys=keys,
                        gradients=tensor.flat_values,
                        embeddings_after=embeddings_after,
                        optimizer_applied=optimizer_applied,
                        gradient_tensor=tensor,
                    )
            traced_gradients = getattr(self, "_traced_gradients", None)
            if optimizer_applied and traced_gradients is not None:
                observer_gradients = traced_gradients
                for dim, gradients_by_key in sorted(observer_gradients.items()):
                    if not gradients_by_key:
                        continue
                    keys = np.asarray(
                        sorted(gradients_by_key), dtype=np.dtype(self.key_dtype)
                    )
                    accumulated_gradients = np.stack(
                        [gradients_by_key[int(key)] for key in keys]
                    )
                    embeddings_after = await self._read_embeddings_for_audit(keys, dim)
                    tracer.sparse_commit(
                        group=f"{self.gname}_dim_{dim}",
                        step=operator_step,
                        spmd_id=f"Embedding_{self.gname}",
                        rank=self.rank,
                        keys=keys,
                        accumulated_gradients=accumulated_gradients,
                        embeddings_after=embeddings_after,
                    )
                observer_gradients.clear()

    async def flush_pending_updates(self) -> None:
        if self._steps_since_commit > 0:
            step = (
                self._current_step
                if self._current_step is not None
                else (self._last_committed_step or 0) + 1
            )
            await self._commit_optimizer_step(step, self._steps_since_commit)
            traced_gradients = getattr(self, "_traced_gradients", None)
            if traced_gradients is not None:
                observer_gradients = traced_gradients
                for dim, gradients_by_key in sorted(observer_gradients.items()):
                    if not gradients_by_key:
                        continue
                    keys = np.asarray(
                        sorted(gradients_by_key), dtype=np.dtype(self.key_dtype)
                    )
                    accumulated_gradients = np.stack(
                        [gradients_by_key[int(key)] for key in keys]
                    )
                    embeddings_after = await self._read_embeddings_for_audit(keys, dim)
                    self.execution_tracer.sparse_commit(
                        group=f"{self.gname}_dim_{dim}",
                        step=step,
                        spmd_id=f"Embedding_{self.gname}",
                        rank=self.rank,
                        keys=keys,
                        accumulated_gradients=accumulated_gradients,
                        embeddings_after=embeddings_after,
                    )
                observer_gradients.clear()

    async def finalize_checkpoint_state(self, _context: Any) -> None:
        await self.flush_pending_updates()


def get_sparse_embedding_class(
    gname: str,
    dims: list[int],
    shards: int,
    resource_profile: Any,
    runtime_env: dict[str, Any] | None,
    ckpt_backend: str | None,
    max_staleness_steps: int,
    optimizer_enabled: bool,
    key_dtype: str,
    load_optimizer_state: bool = True,
    *,
    sparse_partition_assignment_spec: PartitionSpec,
    value_dtype: str,
    gradient_dtype: str,
    gradient_transport_dtype: str,
    optimizer_dtype: str,
) -> type:
    if not isinstance(optimizer_enabled, bool):
        raise TypeError("optimizer_enabled must be bool")
    if not isinstance(load_optimizer_state, bool):
        raise TypeError("load_optimizer_state must be bool")
    if not isinstance(sparse_partition_assignment_spec, PartitionSpec):
        raise TypeError("sparse_partition_assignment_spec must be a PartitionSpec")
    key_dtype = require_key_dtype(key_dtype)
    value_dtype = require_sparse_value_dtype(value_dtype)
    gradient_dtype = require_sparse_gradient_dtype(gradient_dtype)
    gradient_transport_dtype = require_sparse_gradient_transport_dtype(
        gradient_transport_dtype
    )
    optimizer_dtype = require_sparse_optimizer_dtype(optimizer_dtype)
    if not optimizer_enabled and load_optimizer_state:
        raise ValueError("load_optimizer_state=True requires optimizer_enabled=True")
    if not optimizer_enabled and ckpt_backend is not None:
        raise ValueError(
            "optimizer-disabled sparse tables cannot register checkpoint components"
        )
    lookup_async_policy = sparse_lookup_async_policy(max_staleness_steps)
    update_async_policy = sparse_update_async_policy()
    lookup_input_sharding: dict[str, Any] = {}
    lookup_output_sharding: dict[str, Any] = {}
    for d in dims:
        lookup_input_sharding[f"unique_ids_dim{d}"] = {
            "type": "dynamic",
            "axis": ["shard"],
            "strategy": sparse_partition_resource_strategy(f"{gname}.assignment"),
        }
        lookup_output_sharding[f"sparse_features_{gname}_dim_{d}"] = {
            "type": "mirror",
            "reference": f"lookup.unique_ids_dim{d}",
        }

    update_input_sharding: dict[str, Any] = {}
    if optimizer_enabled:
        for d in dims:
            update_input_sharding[f"grads_dim{d}"] = {
                "type": "mirror",
                "reference": f"lookup.sparse_features_{gname}_dim_{d}",
            }
        update_input_sharding["optimizer_update_window"] = {
            "type": "static",
            "axis": [],
        }

    bind_kwargs: dict[str, Any] = {
        "resource_profile": resource_profile,
        "mesh_axis_names": ("shard",),
        "mesh_axis_shapes": (shards,),
        "runtime_env": runtime_env,
        "checkpoint_components": (
            checkpoint_finalizer_registration(
                f"{gname}_embedding_finalize",
                lambda owner, context: owner.finalize_checkpoint_state(context),
            ),
            sparse_checkpoint_registration(
                component_id=f"{gname}_embedding",
                lineage_name=f"{gname}.embedding",
                weight_group=gname,
                backend=ckpt_backend,
                load_optimizer_state=load_optimizer_state,
                mode=CheckpointSaveMode.INCREMENTAL,
                initial_mode=CheckpointSaveMode.FULL,
                include_page_images=False,
            ),
        )
        if ckpt_backend is not None
        else (),
        "lookup": {
            "async_policy": lookup_async_policy,
            "data_argument": [f"unique_ids_dim{d}" for d in dims],
            "input_sharding": lookup_input_sharding,
            "output_sharding": lookup_output_sharding,
        },
    }
    if optimizer_enabled:
        bind_kwargs["update"] = {
            "async_policy": update_async_policy,
            "data_argument": [
                *(f"grads_dim{d}" for d in dims),
                "optimizer_update_window",
            ],
            "input_sharding": update_input_sharding,
        }
    BoundTable = bind_spmd(
        DynamicSparseEmbeddingSPMD,
        **bind_kwargs,
    )
    BoundTable._factory_value_dtype = value_dtype
    BoundTable._factory_key_dtype = key_dtype
    BoundTable._factory_gradient_dtype = gradient_dtype
    BoundTable._factory_gradient_transport_dtype = gradient_transport_dtype
    BoundTable._factory_optimizer_dtype = optimizer_dtype
    BoundTable._factory_assignment_spec = sparse_partition_assignment_spec
    BoundTable._factory_resource_id = f"{gname}.assignment"
    BoundTable._factory_resource_info = {"resource_id": f"{gname}.assignment"}
    BoundTable.__name__ = f"SparseEmbedding_{gname}"
    return BoundTable


__all__ = [
    "DynamicSparseEmbeddingSPMD",
    "create_sparse_optimizer",
    "get_sparse_embedding_class",
    "register_sparse_partition_resource",
    "sparse_lookup_async_policy",
    "sparse_partition_resource_strategy",
    "sparse_update_async_policy",
]
