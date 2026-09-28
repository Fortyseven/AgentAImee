"""Built-in tools for Aimee — all opt-in; the core library ships with none.

Usage:
    from aimee.tools import basic_tools, read
    agent = Aimee(config, tools=basic_tools())             # read, write, edit
    agent = Aimee(config, tools=[read()])                  # just read
    agent = Aimee(config, tools=basic_tools(["read", "bash"]))  # bash is opt-in
"""

from __future__ import annotations

from collections.abc import Iterable

from aimee.tools.base import PathEscapeError, Tool, ToolContext, validate_tools
from aimee.tools.bash import bash
from aimee.tools.fs import edit, read, write

__all__ = [
    "PathEscapeError",
    "Tool",
    "ToolContext",
    "bash",
    "basic_tools",
    "edit",
    "read",
    "validate_tools",
    "write",
]

_FACTORIES: dict[str, type] = {"read": read, "write": write, "edit": edit, "bash": bash}


def basic_tools(names: Iterable[str] | None = None) -> list[Tool]:
    """Fresh instances of the built-in tools.

    `names=None` returns read, write, and edit. `bash` is deliberately NOT in
    the default: it runs arbitrary shell commands and must be added explicitly
    (e.g. `basic_tools(["read", "bash"])` or `bash()`), gated by an
    `on_tool_call` approval hook. A name sequence returns just the selected
    subset in the given order.
    """
    if names is None:
        names = ["read", "write", "edit"]
    tools: list[Tool] = []
    for name in names:
        factory = _FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown basic tool: {name!r} (choose from {sorted(_FACTORIES)})")
        tools.append(factory())
    return tools
