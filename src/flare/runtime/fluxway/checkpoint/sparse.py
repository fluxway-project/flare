from __future__ import annotations

from sparse_embedding.checkpoint import CheckpointSaveMode

from fluxway.integrations.sparse_embedding.checkpoint import (
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
    compact_sparse_checkpoints,
)


def sparse_checkpoint_registration(
    *,
    component_id: str,
    lineage_name: str,
    weight_group: str | None = None,
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
    component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        component_id=component_id,
        weight_group=weight_group,
        lineage_name=lineage_name,
        load_optimizer_state=load_optimizer_state,
        mode=mode,
        initial_mode=initial_mode,
        include_page_images=include_page_images,
    )
    return component.registration()


__all__ = [
    "LocalLanceStorageFactory",
    "SparseEmbeddingCheckpointComponent",
    "compact_sparse_checkpoints",
    "sparse_checkpoint_registration",
]
