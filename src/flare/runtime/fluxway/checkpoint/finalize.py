from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from fluxway.common.types.checkpoint import CheckpointComponentSpec, SaveSession
from fluxway.control_plane.checkpointer import (
    CheckpointComponent,
    CheckpointComponentFinalize,
    CheckpointComponentRegistration,
)

CheckpointFinalizer = Callable[[Any, SaveSession], Awaitable[None] | None]


class CheckpointFinalizerComponent(CheckpointComponent, CheckpointComponentFinalize):
    def __init__(self, component_id: str, finalizer: CheckpointFinalizer) -> None:
        if not callable(finalizer):
            raise TypeError("checkpoint finalizer must be callable")
        self._descriptor = CheckpointComponentSpec(component_id=component_id)
        self._finalizer = finalizer

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self._descriptor

    async def finalize_save(self, session: SaveSession) -> None:
        result = self._finalizer(session.instance, session)
        if inspect.isawaitable(result):
            await result
        elif result is not None:
            raise TypeError("checkpoint finalizer must return None")

    def registration(self) -> CheckpointComponentRegistration:
        return CheckpointComponentRegistration(component=self, finalize=self)


def checkpoint_finalizer_registration(
    component_id: str, finalizer: CheckpointFinalizer
) -> CheckpointComponentRegistration:
    return CheckpointFinalizerComponent(component_id, finalizer).registration()


__all__ = [
    "CheckpointFinalizerComponent",
    "checkpoint_finalizer_registration",
]
