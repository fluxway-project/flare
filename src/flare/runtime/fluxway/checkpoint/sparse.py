from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pyarrow as pa
from sparse_embedding.checkpoint import CheckpointSaveMode

from fluxway.integrations.sparse_embedding.checkpoint import (
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
    compact_sparse_checkpoints,
)

_ResultT = TypeVar("_ResultT")


async def _retry_arrow_invalid(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    attempts: int = 3,
    initial_delay: float = 0.2,
) -> _ResultT:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return await operation()
        except pa.ArrowInvalid:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(initial_delay * (attempt + 1))
    raise AssertionError("retry loop did not return or raise")


class _RetryingSparseCheckpointComponent(SparseEmbeddingCheckpointComponent):
    """Retry transient Lance manifest reads after a concurrent publication."""

    async def publish_save(self, session) -> None:
        parent_operation = super().publish_save
        await _retry_arrow_invalid(lambda: parent_operation(session))

    async def acknowledge_save(self, permit) -> None:
        parent_operation = super().acknowledge_save
        await _retry_arrow_invalid(lambda: parent_operation(permit))


def sparse_checkpoint_registration(
    *,
    component_id: str,
    lineage_name: str,
    output_path: str | None = None,
    backend: str,
    load_optimizer_state: bool = True,
    mode: CheckpointSaveMode,
    initial_mode: CheckpointSaveMode,
    include_page_images: bool = False,
):
    if backend != "lance":
        raise ValueError(
            f"unsupported sparse checkpoint backend {backend!r}; expected 'lance'"
        )
    weight_group = None
    if output_path is not None:
        prefix, separator, weight_group = output_path.partition("/")
        if (
            prefix != "sparse"
            or not separator
            or not weight_group
            or "/" in weight_group
        ):
            raise ValueError("sparse checkpoint output_path must be sparse/<group>")
    # Fluxway's current checkpoint component names this field ``output_path``;
    # older installations used ``weight_group``.  Select by signature so the
    # application remains compatible while the runtime is upgraded in place.
    component_kwargs = {
        "component_id": component_id,
        "lineage_name": lineage_name,
        "load_optimizer_state": load_optimizer_state,
        "mode": mode,
        "initial_mode": initial_mode,
        "include_page_images": include_page_images,
    }
    parameters = inspect.signature(_RetryingSparseCheckpointComponent).parameters
    if "output_path" in parameters:
        component_kwargs["output_path"] = output_path
    elif "weight_group" in parameters:
        component_kwargs["weight_group"] = weight_group
    else:
        raise TypeError(
            "SparseEmbeddingCheckpointComponent exposes neither output_path "
            "nor weight_group"
        )
    component = _RetryingSparseCheckpointComponent(
        LocalLanceStorageFactory(), **component_kwargs
    )
    return component.registration()


__all__ = [
    "LocalLanceStorageFactory",
    "SparseEmbeddingCheckpointComponent",
    "compact_sparse_checkpoints",
    "sparse_checkpoint_registration",
]
