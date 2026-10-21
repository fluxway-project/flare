"""Invoke synchronous or asynchronous source ports without blocking asyncio."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


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
