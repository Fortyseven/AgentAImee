"""Tool primitives: the Tool spec and the ToolContext passed to handlers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aimee.config import AimeeConfig

# A handler may be sync or async and must return a string (or str-convertible).
Handler = Callable[[dict[str, Any], "ToolContext"], Any]


class PathEscapeError(ValueError):
    """A tool path resolved outside the workspace roots; containment rejected it."""


@dataclass
class ToolContext:
    """Workspace view handed to every tool handler (built-in and custom)."""

    config: AimeeConfig
    roots: list[Path] = field(default_factory=list)

    @property
    def primary_root(self) -> Path:
        return self.roots[0]

    def _in_workspace(self, candidate: Path) -> bool:
        """True if candidate stays inside a workspace root after full resolution.

        `Path.resolve()` collapses `..` and follows symlinks, so traversal and
        symlink escapes are both caught. No-op when `allow_path_escape` is set.
        """
        if self.config.allow_path_escape:
            return True
        target = candidate.resolve()
        for root in self.roots:
            try:
                if target.is_relative_to(root.resolve()):
                    return True
            except ValueError:
                continue  # different drive (Windows) — not contained
        return False

    def _reject_escape(self, path: str | Path, candidate: Path) -> None:
        raise PathEscapeError(
            f"Path escapes workspace roots: {path!r} resolves to {candidate.resolve()}; "
            "set AimeeConfig(allow_path_escape=True) to permit"
        )

    def resolve(self, path: str | Path, *, must_exist: bool) -> Path:
        """Resolve a path against the workspace roots.

        Relative paths are tried against each root in order: for reads the
        first root where the file exists wins; for writes the first root that
        contains it wins, else the primary root is used. Absolute paths are
        used as-is. With `allow_path_escape` off (default), any path that
        resolves outside the roots — via `..` traversal, a symlink, or an
        absolute path — raises PathEscapeError.
        """
        p = Path(path)
        candidates = [p] if p.is_absolute() else [root / p for root in self.roots]
        for candidate in candidates:
            if self._in_workspace(candidate) and candidate.exists():
                return candidate
        if must_exist:
            escaped = next((c for c in candidates if not self._in_workspace(c)), None)
            if escaped is not None:
                self._reject_escape(path, escaped)
            roots_str = ", ".join(str(r) for r in self.roots)
            raise FileNotFoundError(f"'{path}' not found under workspace roots: {roots_str}")
        target = candidates[0]
        if not self._in_workspace(target):
            self._reject_escape(path, target)
        return target


@dataclass
class Tool:
    """One named tool: OpenAI function schema + handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def to_openai(self) -> dict[str, Any]:
        """OpenAI function-calling schema for this tool."""
        params = self.parameters or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


def validate_tools(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Index tools by name; raises on duplicates or invalid names."""
    by_name: dict[str, Tool] = {}
    for tool in tools:
        if not tool.name.isidentifier():
            raise ValueError(f"Invalid tool name (must be a Python identifier): {tool.name!r}")
        if tool.name in by_name:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        by_name[tool.name] = tool
    return by_name
