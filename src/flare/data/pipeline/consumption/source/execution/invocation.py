"""Invoke synchronous or asynchronous source ports without blocking asyncio."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from fluxway.common.ingestion import SourceWorkItem

from flare.data.pipeline.consumption.source.contracts.source import (
    SourceBatch,
    SourceReadSession,
)


async def invoke_source(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke one source method without running blocking code on the event loop."""
    method_call = method.__call__ if callable(method) else None
    if inspect.iscoroutinefunction(method) or inspect.iscoroutinefunction(method_call):
        result = method(*args, **kwargs)
    else:
        result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def read_source_batch(
    session: SourceReadSession, step: int, *, work_items: Sequence[SourceWorkItem]
) -> SourceBatch:
    """Validate scheduling facts without interpreting a backend task token."""
    if type(step) is not int or step < 0:
        raise ValueError("source step must be a non-negative integer")
    if not work_items:
        raise ValueError("source read requires acquired work items")
    cardinality = 0
    for item in work_items:
        if not isinstance(item, SourceWorkItem):
            raise TypeError("source read requires SourceWorkItem values")
        if item.key.step != step:
            raise ValueError("source work item step does not match the requested step")
        cardinality += item.cardinality
    result = await invoke_source(session.read, step, work_items=work_items)
    if not isinstance(result, SourceBatch):
        raise TypeError("source read must return SourceBatch")
    if result.cardinality != cardinality:
        raise ValueError("source batch cardinality differs from acquired source work")
    return result
