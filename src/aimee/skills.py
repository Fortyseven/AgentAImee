"""Skill discovery: SKILL.md files with name/description frontmatter.

A skill is a directory containing a ``SKILL.md`` with minimal YAML frontmatter::

    ---
    name: my-skill
    description: When and how to use this skill.
    ---
    Full markdown body (read by the agent on demand).

Only flat ``key: value`` frontmatter lines are parsed (no PyYAML dependency).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)


@dataclass
class Skill:
    """One discovered skill: catalog metadata plus the on-demand body."""

    name: str
    description: str
    path: Path
    body: str = ""


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split minimal YAML frontmatter from a markdown document.

    Returns (metadata, body). No frontmatter → ({}, original text).
    """
    meta: dict[str, str] = {}
    match = _FRONTMATTER.match(text)
    if not match:
        return meta, text
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[match.end() :]


def load_skill(path: Path) -> Skill | None:
    """Load one SKILL.md; returns None when unreadable or empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = parse_frontmatter(text)
    name = meta.get("name") or path.parent.name
    return Skill(name=name, description=meta.get("description", ""), path=path, body=body.strip())


def load_skills(dirs: Sequence[Path]) -> list[Skill]:
    """Scan immediate subdirectories of each dir for a SKILL.md.

    Missing dirs are skipped. For duplicate skill names, the first one
    (in directory order) wins.
    """
    skills: list[Skill] = []
    seen: set[str] = set()
    for directory in dirs:
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            skill_file = entry / "SKILL.md"
            if not (entry.is_dir() and skill_file.is_file()):
                continue
            skill = load_skill(skill_file)
            if skill is not None and skill.name not in seen:
                seen.add(skill.name)
                skills.append(skill)
    return skills


def render_skills_catalog(skills: Sequence[Skill]) -> str:
    """Render the system-prompt catalog block (metadata only, not bodies)."""
    lines = [
        "# Skills",
        "",
        "Available skills. Before using one, read its SKILL.md with the read tool and follow it.",
        "",
    ]
    for skill in skills:
        suffix = f" — {skill.description}" if skill.description else ""
        lines.append(f"- {skill.name}{suffix}")
        lines.append(f"  {skill.path}")
    return "\n".join(lines)
