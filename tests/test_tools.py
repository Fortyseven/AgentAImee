"""Built-in tool tests: read, write, edit, bash, multi-root resolution, basic_tools()."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from aimee import AimeeConfig
from aimee.tools import PathEscapeError, Tool, bash, basic_tools, edit, read, write
from aimee.tools.base import ToolContext


def make_ctx(tmp_path: Path, *extra_roots: Path, **cfg) -> ToolContext:
    config = AimeeConfig(roots=[tmp_path, *extra_roots], **cfg)
    return ToolContext(config=config, roots=config.resolved_roots())


def call(tool, args: dict, ctx: ToolContext) -> str:
    return tool.handler(args, ctx)


# -- read --------------------------------------------------------------------


def test_read_numbered_lines(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree", encoding="utf-8")
    out = call(read(), {"path": "a.txt"}, make_ctx(tmp_path))
    assert out == "     1\tone\n     2\ttwo\n     3\tthree"


def test_read_offset_and_limit(tmp_path):
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 6)), encoding="utf-8")
    out = call(read(), {"path": "a.txt", "offset": 2, "limit": 2}, make_ctx(tmp_path))
    assert "     2\tline2" in out
    assert "     3\tline3" in out
    assert "line4" not in out
    assert "pass offset=4" in out


def test_read_past_end(tmp_path):
    (tmp_path / "a.txt").write_text("only", encoding="utf-8")
    out = call(read(), {"path": "a.txt", "offset": 99}, make_ctx(tmp_path))
    assert "past end" in out


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found under workspace roots"):
        call(read(), {"path": "nope.txt"}, make_ctx(tmp_path))


def test_read_directory_lists(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    out = call(read(), {"path": "."}, make_ctx(tmp_path))
    assert "f.txt" in out
    assert "sub/" in out


# -- read: resource guards (TODO #4) ------------------------------------------


def test_read_fifo_rejected(tmp_path):
    # A FIFO with no writer blocks open() forever; must be refused up front.
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="not a regular file"):
        call(read(), {"path": "pipe"}, make_ctx(tmp_path))


def test_read_device_rejected(tmp_path):
    if not Path("/dev/urandom").exists():
        pytest.skip("no /dev/urandom on this platform")
    ctx = make_ctx(tmp_path, allow_path_escape=True)
    with pytest.raises(ValueError, match="not a regular file"):
        call(read(), {"path": "/dev/urandom"}, ctx)


def _big_file(tmp_path: Path, n: int) -> None:
    (tmp_path / "big.txt").write_text(
        "\n".join(f"line{i}" for i in range(1, n + 1)), encoding="utf-8"
    )


def test_read_huge_file_deep_offset(tmp_path):
    n = 200_000
    _big_file(tmp_path, n)
    # A deep offset must reach the last lines without loading the whole file
    # into memory; a window ending at EOF gets no continuation footer.
    out = call(read(), {"path": "big.txt", "offset": n - 1, "limit": 2}, make_ctx(tmp_path))
    assert f"{n - 1}\tline{n - 1}" in out
    assert f"{n}\tline{n}" in out
    assert "file continues" not in out


def test_read_huge_file_past_end(tmp_path):
    _big_file(tmp_path, 200_000)
    out = call(read(), {"path": "big.txt", "offset": 999_999, "limit": 10}, make_ctx(tmp_path))
    assert "offset 999999 is past end (200000 lines)" in out


def test_read_huge_file_bounded_window_footer(tmp_path):
    _big_file(tmp_path, 200_000)
    # Window in the middle: scan stops after the window, total is a lower bound.
    out = call(read(), {"path": "big.txt", "offset": 10, "limit": 5}, make_ctx(tmp_path))
    assert "line10" in out and "line14" in out
    assert "line15" not in out
    assert "the file continues" in out
    assert "pass offset=15" in out


def test_read_oversized_line_rejected(tmp_path, monkeypatch):
    import aimee.tools.fs as fsmod

    monkeypatch.setattr(fsmod, "_MAX_LINE_BYTES", 100)
    (tmp_path / "blob.txt").write_text("a" * 200 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        call(read(), {"path": "blob.txt"}, make_ctx(tmp_path))


# -- write --------------------------------------------------------------------


def test_write_creates_parents_and_content(tmp_path):
    out = call(write(), {"path": "deep/nested/new.txt", "content": "hello"}, make_ctx(tmp_path))
    assert (tmp_path / "deep/nested/new.txt").read_text(encoding="utf-8") == "hello"
    assert "Wrote 5 chars" in out


def test_write_overwrites(tmp_path):
    (tmp_path / "a.txt").write_text("old", encoding="utf-8")
    call(write(), {"path": "a.txt", "content": "new"}, make_ctx(tmp_path))
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new"


# -- edit ---------------------------------------------------------------------


def test_edit_unique_match(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\ngamma", encoding="utf-8")
    call(edit(), {"path": "a.txt", "old_text": "beta", "new_text": "BETA"}, make_ctx(tmp_path))
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\nBETA\ngamma"


def test_edit_no_match(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        call(edit(), {"path": "a.txt", "old_text": "zzz", "new_text": "x"}, make_ctx(tmp_path))


def test_edit_multiple_matches_rejected(tmp_path):
    (tmp_path / "a.txt").write_text("dup dup", encoding="utf-8")
    with pytest.raises(ValueError, match="matches 2 times"):
        call(edit(), {"path": "a.txt", "old_text": "dup", "new_text": "x"}, make_ctx(tmp_path))


def test_edit_empty_old_text_rejected(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        call(edit(), {"path": "a.txt", "old_text": "", "new_text": "x"}, make_ctx(tmp_path))


# -- bash ---------------------------------------------------------------------


def test_bash_echo_and_exit_code(tmp_path):
    out = asyncio.run(bash().handler({"command": "echo hi"}, make_ctx(tmp_path)))
    assert out.startswith("[exit 0]")
    assert "hi" in out


def test_bash_nonzero_exit(tmp_path):
    out = asyncio.run(bash().handler({"command": "exit 3"}, make_ctx(tmp_path)))
    assert out.startswith("[exit 3]")


def test_bash_runs_in_primary_root(tmp_path):
    (tmp_path / "marker.txt").write_text("m", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "marker.txt").write_text("o", encoding="utf-8")
    ctx = make_ctx(tmp_path, other)
    out = asyncio.run(bash().handler({"command": "cat marker.txt"}, ctx))
    assert "m" in out and "o" not in out


def test_bash_timeout(tmp_path):
    ctx = make_ctx(tmp_path, bash_timeout=1)
    out = asyncio.run(bash().handler({"command": "sleep 5"}, ctx))
    assert out == "[exit -1] Command timed out after 1s"


def test_bash_truncates_output(tmp_path):
    ctx = make_ctx(tmp_path, output_limit=10)
    out = asyncio.run(bash().handler({"command": "printf 'A'%.0s $(seq 1 50)"}, ctx))
    assert "[output truncated to 10 chars]" in out
    assert len(out) < 100


# -- multi-root resolution ------------------------------------------------------


def test_multi_root_read_finds_file_in_secondary(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "shared.txt").write_text("secondary content", encoding="utf-8")
    ctx = make_ctx(tmp_path, other)
    out = call(read(), {"path": "shared.txt"}, ctx)
    assert "secondary content" in out


def test_multi_root_read_prefers_primary(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "same.txt").write_text("primary", encoding="utf-8")
    (other / "same.txt").write_text("secondary", encoding="utf-8")
    ctx = make_ctx(tmp_path, other)
    out = call(read(), {"path": "same.txt"}, ctx)
    assert "primary" in out


def test_multi_root_write_new_file_goes_to_primary(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    ctx = make_ctx(tmp_path, other)
    call(write(), {"path": "brand-new.txt", "content": "x"}, ctx)
    assert (tmp_path / "brand-new.txt").exists()
    assert not (other / "brand-new.txt").exists()


def test_multi_root_write_existing_file_updates_in_place(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("old", encoding="utf-8")
    ctx = make_ctx(tmp_path, other)
    call(write(), {"path": "keep.txt", "content": "new"}, ctx)
    assert (other / "keep.txt").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "keep.txt").exists()


def test_multi_root_edit_across_roots(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "doc.md").write_text("hello", encoding="utf-8")
    ctx = make_ctx(tmp_path, other)
    call(edit(), {"path": "doc.md", "old_text": "hello", "new_text": "goodbye"}, ctx)
    assert (other / "doc.md").read_text(encoding="utf-8") == "goodbye"


def test_absolute_path_outside_rejected_by_default(tmp_path):
    target = Path("/tmp/aimee-absolute-test.txt")
    with pytest.raises(PathEscapeError, match="escapes workspace"):
        call(write(), {"path": str(target), "content": "abs"}, make_ctx(tmp_path))
    assert not target.exists()


# -- workspace containment ----------------------------------------------------


def make_nested_ctx(tmp_path: Path, **cfg) -> tuple[ToolContext, Path]:
    """ctx rooted at tmp_path/repo, with a secret file as its sibling."""
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    config = AimeeConfig(roots=[repo], **cfg)
    return ToolContext(config=config, roots=config.resolved_roots()), secret


def test_read_dotdot_traversal_rejected(tmp_path):
    ctx, _ = make_nested_ctx(tmp_path)
    with pytest.raises(PathEscapeError, match="escapes workspace"):
        call(read(), {"path": "../secret.txt"}, ctx)


def test_write_dotdot_traversal_rejected(tmp_path):
    ctx, secret = make_nested_ctx(tmp_path)
    with pytest.raises(PathEscapeError):
        call(write(), {"path": "../secret.txt", "content": "pwned"}, ctx)
    assert secret.read_text(encoding="utf-8") == "top secret"


def test_write_absolute_outside_rejected(tmp_path):
    ctx, secret = make_nested_ctx(tmp_path)
    with pytest.raises(PathEscapeError):
        call(write(), {"path": str(secret), "content": "pwned"}, ctx)
    assert secret.read_text(encoding="utf-8") == "top secret"


def test_edit_outside_rejected(tmp_path):
    ctx, secret = make_nested_ctx(tmp_path)
    with pytest.raises(PathEscapeError):
        call(edit(), {"path": str(secret), "old_text": "top", "new_text": "nope"}, ctx)


def test_read_symlink_escape_rejected(tmp_path):
    ctx, secret = make_nested_ctx(tmp_path)
    (ctx.primary_root / "sneaky").symlink_to(secret)
    with pytest.raises(PathEscapeError):
        call(read(), {"path": "sneaky"}, ctx)


def test_read_absolute_inside_workspace_allowed(tmp_path):
    ctx, _ = make_nested_ctx(tmp_path)
    (ctx.primary_root / "a.txt").write_text("inside", encoding="utf-8")
    out = call(read(), {"path": str(ctx.primary_root / "a.txt")}, ctx)
    assert "inside" in out


def test_allow_path_escape_permits_traversal(tmp_path):
    ctx, secret = make_nested_ctx(tmp_path, allow_path_escape=True)
    out = call(read(), {"path": "../secret.txt"}, ctx)
    assert "top secret" in out
    call(write(), {"path": "../secret.txt", "content": "pwned"}, ctx)
    assert secret.read_text(encoding="utf-8") == "pwned"


def test_allow_path_escape_permits_absolute(tmp_path):
    ctx = make_ctx(tmp_path, allow_path_escape=True)
    target = Path("/tmp/aimee-absolute-test.txt")
    try:
        call(write(), {"path": str(target), "content": "abs"}, ctx)
        assert target.read_text(encoding="utf-8") == "abs"
    finally:
        target.unlink(missing_ok=True)


# -- basic_tools() --------------------------------------------------------------


def test_basic_tools_all():
    tools = basic_tools()
    assert [t.name for t in tools] == ["read", "write", "edit"]
    assert all(isinstance(t, Tool) for t in tools)


def test_basic_tools_default_excludes_bash():
    # bash runs arbitrary shell commands; it must be an explicit opt-in.
    assert "bash" not in [t.name for t in basic_tools()]


def test_basic_tools_subset_in_order():
    tools = basic_tools(["bash", "read"])
    assert [t.name for t in tools] == ["bash", "read"]


def test_basic_tools_fresh_instances():
    a, b = basic_tools(["read"]), basic_tools(["read"])
    assert a[0] is not b[0]  # new instances, not shared singletons


def test_basic_tools_unknown_name():
    with pytest.raises(ValueError, match="Unknown basic tool"):
        basic_tools(["nope"])
