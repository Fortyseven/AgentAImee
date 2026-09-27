"""System prompt assembly: base prompt + AGENTS.md + skills catalog."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aimee.config import AimeeConfig
from aimee.skills import Skill, render_skills_catalog

DEFAULT_SYSTEM_PROMPT = """\
You are Aimee, a minimal LLM agent embedded in the client application.

You accomplish tasks by calling tools. Rules of thumb:
- Prefer small, verifiable steps; check results (e.g. re-read a file after editing) when cheap.
- Use relative paths; they resolve against the workspace roots.
- When the task is complete, reply with a concise final answer and stop calling tools.
"""

AGENTS_MD_NAME = "AGENTS.md"


def find_agents_md(roots: Sequence[Path]) -> Path | None:
    """Find the nearest AGENTS.md walking up from each root, in root order."""
    for root in roots:
        current = root
        while True:
            candidate = current / AGENTS_MD_NAME
            if candidate.is_file():
                return candidate
            if current.parent == current:
                break
            current = current.parent
    return None


def build_system_prompt(config: AimeeConfig, skills: Sequence[Skill]) -> str:
    """Assemble the full system prompt for a run."""
    parts: list[str] = [config.system_prompt or DEFAULT_SYSTEM_PROMPT]

    agents_md: Path | None = None
    if config.agents_md is not None:
        agents_md = config.agents_md
    else:
        agents_md = find_agents_md(config.resolved_roots())
    if agents_md is not None and agents_md.is_file():
        body = agents_md.read_text(encoding="utf-8").strip()
        parts.append(f"# Project instructions ({agents_md})\n\n{body}")

    if skills:
        parts.append(render_skills_catalog(skills))
    return "\n\n".join(parts)
