"""System prompt assembly tests: base prompt, AGENTS.md discovery, skills catalog."""

from __future__ import annotations

from pathlib import Path

from aimee.config import AimeeConfig
from aimee.prompt import build_system_prompt, find_agents_md
from aimee.skills import Skill


def test_default_base_prompt(tmp_path):
    prompt = build_system_prompt(AimeeConfig(roots=[tmp_path]), skills=[])
    assert "Aimee" in prompt
    assert "AGENTS.md" not in prompt  # none present
    assert "# Skills" not in prompt


def test_custom_system_prompt_replaces_base(tmp_path):
    config = AimeeConfig(roots=[tmp_path], system_prompt="You are a test bot.")
    prompt = build_system_prompt(config, skills=[])
    assert prompt.startswith("You are a test bot.")
    assert "You are Aimee" not in prompt


def test_explicit_agents_md_included(tmp_path):
    agents = tmp_path / "MY_RULES.md"
    agents.write_text("rule one", encoding="utf-8")
    config = AimeeConfig(roots=[tmp_path], agents_md=agents)
    prompt = build_system_prompt(config, skills=[])
    assert "rule one" in prompt
    assert str(agents) in prompt


def test_agents_md_discovered_walking_up(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("top rules", encoding="utf-8")
    config = AimeeConfig(roots=[nested])
    prompt = build_system_prompt(config, skills=[])
    assert "top rules" in prompt


def test_agents_md_first_root_order(tmp_path):
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    r1.mkdir()
    r2.mkdir()
    (r2 / "AGENTS.md").write_text("second root rules", encoding="utf-8")
    config = AimeeConfig(roots=[r1, r2])
    assert find_agents_md([r1, r2]) == r2 / "AGENTS.md"
    prompt = build_system_prompt(config, skills=[])
    assert "second root rules" in prompt


def test_missing_explicit_agents_md_is_ignored(tmp_path):
    config = AimeeConfig(roots=[tmp_path], agents_md=tmp_path / "nope.md")
    prompt = build_system_prompt(config, skills=[])
    assert "nope.md" not in prompt


def test_skills_catalog_in_prompt(tmp_path):
    skills = [
        Skill(
            name="demo",
            description="A demo skill.",
            path=Path("/tmp/skills/demo/SKILL.md"),
            body="irrelevant for the catalog",
        )
    ]
    prompt = build_system_prompt(AimeeConfig(roots=[tmp_path]), skills=skills)
    assert "# Skills" in prompt
    assert "- demo — A demo skill." in prompt
    assert "/tmp/skills/demo/SKILL.md" in prompt
    assert "read" in prompt  # instructs the agent to read before use
