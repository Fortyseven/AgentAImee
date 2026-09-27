"""Built-in tools for Aimee — all opt-in; the core library ships with none.

Usage:
    from aimee.tools import basic_tools, read
    agent = Aimee(config, tools=basic_tools())             # read, write, edit, bash
    agent = Aimee(config, tools=[read()])                  # just read
    agent = Aimee(config, tools=basic_tools(["read", "edit"]))
"""

from __future__ import annotations

from collections.abc import Iterable

from aimee.tools.base import Tool, ToolContext, validate_tools
from aimee.tools.bash import bash
from aimee.tools.fs import edit, read, write

__all__ = ["Tool", "ToolContext", "bash", "basic_tools", "edit", "read", "validate_tools", "write"]

_FACTORIES: dict[str, type] = {"read": read, "write": write, "edit": edit, "bash": bash}


def basic_tools(names: Iterable[str] | None = None) -> list[Tool]:
    """Fresh instances of the built-in tools.

    `names=None` returns all four (read, write, edit, bash); a name sequence
    returns just the selected subset in the given order.
    """
    if names is None:
        names = ["read", "write", "edit", "bash"]
    tools: list[Tool] = []
    for name in names:
        factory = _FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown basic tool: {name!r} (choose from {sorted(_FACTORIES)})")
        tools.append(factory())
    return tools
