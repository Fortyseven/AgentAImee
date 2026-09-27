"""Callback hooks for extending Aimee.

Hooks are duck-typed: register any object exposing any subset of the methods
below (sync or async). The agent calls a hook only for methods it defines:

- ``on_delta(chunk: str) -> None``
    One streamed chunk of assistant text.
- ``on_tool_call(name: str, args: dict) -> ToolDecision | None``
    Approval gate before a tool runs. Return None (or don't define the
    method) to allow, ``ToolDecision.deny(reason)`` to block, or
    ``ToolDecision.modify(new_args)`` to replace the arguments. The first
    hook returning a non-None decision wins.
- ``on_tool_result(name: str, args: dict, result: str) -> None``
    After a tool finished (including denials and errors).
- ``on_turn(turn: int, response: ChatResponse) -> None``
    After each completed model response.
- ``on_error(error: Exception) -> None``
    On tool execution errors (run continues) and API errors (run aborts).
- ``on_done(report: RunReport) -> None``
    Once, when the run finishes.
"""

from __future__ import annotations

import inspect
from typing import Any


async def call_hook(hook: Any, method: str, *args: Any) -> Any:
    """Call ``hook.method(*args)`` if defined; supports sync and async methods.

    Returns None when the hook doesn't define the method.
    """
    fn = getattr(hook, method, None)
    if fn is None:
        return None
    result = fn(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def fire(hooks: list[Any], method: str, *args: Any) -> None:
    """Call ``method`` on every hook that defines it, in registration order."""
    for hook in hooks:
        await call_hook(hook, method, *args)
