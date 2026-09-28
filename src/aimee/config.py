"""Configuration for an Aimee agent. The client owns all of these values."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_API_BASE = "OPENAI_API_BASE"
ENV_API_KEY = "OPENAI_API_KEY"
DEFAULT_API_BASE = "https://api.openai.com/v1"


def _default_roots() -> list[Path]:
    return [Path.cwd()]


@dataclass
class AimeeConfig:
    """Everything Aimee needs from the client. All fields are optional.

    `roots` is the list of workspace roots (in priority order). `roots[0]` is
    the primary root: relative paths resolve against roots in order (first
    existing file wins for read/edit), new files are created under the first
    root, bash runs with the first root as cwd, and AGENTS.md discovery walks
    up from each root in order until one is found.
    """

    roots: list[Path] = field(default_factory=_default_roots)
    model: str = "qwen38/27b-default"
    api_base: str | None = None
    api_key: str | None = None
    verify_tls: bool = True
    system_prompt: str | None = None
    agents_md: Path | None = None
    skills_dirs: list[Path] = field(default_factory=list)
    max_turns: int = 30
    temperature: float | None = None
    max_tokens: int | None = None
    bash_timeout: int = 120
    output_limit: int = 100_000

    def resolved_api_base(self) -> str:
        """API base URL: explicit config, else $OPENAI_API_BASE, else OpenAI."""
        return self.api_base or os.environ.get(ENV_API_BASE) or DEFAULT_API_BASE

    def resolved_api_key(self) -> str:
        """API key: explicit config, else $OPENAI_API_KEY. Raises if unset."""
        key = self.api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise ValueError(f"No API key: set {ENV_API_KEY} or pass AimeeConfig(api_key=...)")
        return key

    def resolved_roots(self) -> list[Path]:
        """Expanded, absolute workspace roots (first entry is the primary root)."""
        if not self.roots:
            return _default_roots()
        return [Path(p).expanduser().absolute() for p in self.roots]
