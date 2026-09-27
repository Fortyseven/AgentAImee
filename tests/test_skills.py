"""Skill discovery tests: frontmatter parsing, scanning, catalog rendering."""

from __future__ import annotations

from pathlib import Path

from aimee.skills import load_skill, load_skills, parse_frontmatter, render_skills_catalog


def write_skill(
    base: Path, name: str, *, description: str | None = "desc", body: str = "Body.", **extra
) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    fields: dict[str, str] = {"name": name}
    if description is not None:
        fields["description"] = description
    fields.update(extra)
    text = "---\n" + "\n".join(f"{k}: {v}" for k, v in fields.items()) + "\n---\n\n" + body
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d


def test_parse_frontmatter_valid():
    meta, body = parse_frontmatter("---\nname: x\ndescription: y\n---\n\nHello body\n")
    assert meta == {"name": "x", "description": "y"}
    assert body.strip() == "Hello body"


def test_parse_frontmatter_missing_description():
    meta, body = parse_frontmatter("---\nname: x\n---\nBody\n")
    assert meta == {"name": "x"}
    assert "description" not in meta


def test_parse_frontmatter_absent():
    meta, body = parse_frontmatter("Just a doc\n")
    assert meta == {}
    assert body == "Just a doc\n"


def test_parse_frontmatter_quoted_values():
    meta, _ = parse_frontmatter("---\nname: \"quoted name\"\ndescription: 'single'\n---\n")
    assert meta == {"name": "quoted name", "description": "single"}


def test_load_skill_falls_back_to_dir_name(tmp_path):
    d = tmp_path / "implicit-name"
    d.mkdir()
    (d / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
    skill = load_skill(d / "SKILL.md")
    assert skill is not None
    assert skill.name == "implicit-name"
    assert skill.description == ""


def test_load_skill_unreadable_returns_none(tmp_path):
    assert load_skill(tmp_path / "ghost/SKILL.md") is None


def test_load_skills_scans_immediate_subdirs_only(tmp_path):
    write_skill(tmp_path, "alpha")
    write_skill(tmp_path, "beta", description="second")
    (tmp_path / "loose.md").write_text("not a skill", encoding="utf-8")
    nested = tmp_path / "gamma" / "delta"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: delta\n---\n", encoding="utf-8"
    )  # too deep, ignored

    skills = load_skills([tmp_path])
    assert [s.name for s in skills] == ["alpha", "beta"]
    assert skills[1].description == "second"
    assert skills[0].body == "Body."


def test_load_skills_multiple_dirs_and_duplicates(tmp_path):
    d1, d2 = tmp_path / "s1", tmp_path / "s2"
    d1.mkdir()
    d2.mkdir()
    write_skill(d1, "dup", description="first")
    write_skill(d2, "dup", description="second")
    write_skill(d2, "only-second")

    skills = load_skills([d1, d2, tmp_path / "missing"])
    assert [s.name for s in skills] == ["dup", "only-second"]
    assert skills[0].description == "first"  # first dir wins


def test_render_catalog():
    from aimee.skills import Skill

    text = render_skills_catalog(
        [
            Skill(name="a", description="Does A.", path=Path("/s/a/SKILL.md")),
            Skill(name="b", description="", path=Path("/s/b/SKILL.md")),
        ]
    )
    assert text.splitlines()[0] == "# Skills"
    assert "- a — Does A." in text
    assert "- b\n" in text  # no description → no dash suffix
    assert "/s/a/SKILL.md" in text
