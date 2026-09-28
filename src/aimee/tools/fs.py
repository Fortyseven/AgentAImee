"""Built-in filesystem tools: read, write, edit. All opt-in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aimee.tools.base import Tool

DEFAULT_READ_LIMIT = 2000
_READ_CHUNK_SIZE = 1 << 20   # bytes scanned per iteration
_MAX_LINE_BYTES = 10_000_000  # a line longer than this is treated as binary junk


def _read_window(path: Path, offset: int, limit: int) -> tuple[list[str], int, bool]:
    """Stream a file and return (window, total, complete).

    `window` holds the 1-based lines [offset, offset+limit). Reads in chunks
    and keeps only that window, so memory is O(window + one line) instead of
    O(file), and scanning stops as soon as the window is satisfied — at which
    point `complete` is False and `total` is only a lower bound. Lines are
    split on \\n with a trailing \\r stripped, so \\r\\n files count like
    str.splitlines(). A line longer than _MAX_LINE_BYTES raises ValueError
    rather than growing unbounded.
    """
    window: list[str] = []
    pending = b""  # trailing partial line carried over from the previous chunk
    total = 0
    want_lo = offset
    want_hi = offset + limit

    def keep(line_no: int, raw: bytes) -> None:
        if want_lo <= line_no < want_hi:
            window.append(raw.rstrip(b"\r").decode("utf-8", errors="replace"))

    with path.open("rb") as f:
        while True:
            chunk = f.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            parts = (pending + chunk).split(b"\n")
            pending = parts.pop()
            for part in parts:
                total += 1
                too_long = len(part) > _MAX_LINE_BYTES
                if too_long or len(pending) > _MAX_LINE_BYTES:
                    line_no = total if too_long else total + 1
                    raise ValueError(
                        f"'{path}': line {line_no} exceeds {_MAX_LINE_BYTES} bytes; "
                        "refusing to read (file looks binary or pathological)"
                    )
                keep(total, part)
            if total >= want_hi:
                return window, total, False
        if pending:  # final line without a trailing newline
            total += 1
            keep(total, pending)
        return window, total, True


def read() -> Tool:
    """File reader: numbered lines with offset/limit paging."""

    def handler(args: dict[str, Any], ctx: Any) -> str:
        path = ctx.resolve(args["path"], must_exist=True)
        if path.is_dir():
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in path.iterdir())
            return "\n".join(entries) if entries else "[empty directory]"
        # Devices (/dev/urandom), FIFOs, and sockets block or grow without
        # bound — only plain files are readable, no matter how deep the offset.
        if not path.is_file():
            raise ValueError(
                f"'{path}' is not a regular file; read supports only plain files "
                "and directories (refusing devices, FIFOs, and sockets)"
            )
        offset = max(int(args.get("offset", 1)), 1)
        limit = max(int(args.get("limit", DEFAULT_READ_LIMIT)), 1)
        chunk, total, complete = _read_window(path, offset, limit)
        out = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
        if not chunk:
            out += f"\n[empty file, or offset {offset} is past end ({total} lines)]"
        else:
            # An exact total only exists after scanning to EOF — which means the
            # window already reaches the end — so a continuation footer is only
            # ever paired with a "file continues" (lower-bound) note.
            if not complete:
                shown_to = offset - 1 + len(chunk)
                out += f"\n[showing lines {offset}-{shown_to}; the file continues; "
                out += f"pass offset={shown_to + 1} to read more]"
        return out

    return Tool(
        name="read",
        description=(
            "Read a text file (numbered lines) or list a directory. "
            "Relative paths resolve against the workspace roots. "
            "Use offset/limit to page through long files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path"},
                "offset": {
                    "type": "integer",
                    "description": "First line to show (1-based, default 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines (default {DEFAULT_READ_LIMIT})",
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


def write() -> Tool:
    """File writer: creates parents, overwrites existing files."""

    def handler(args: dict[str, Any], ctx: Any) -> str:
        path = ctx.resolve(args["path"], must_exist=False)
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    return Tool(
        name="write",
        description="Write (create or overwrite) a text file. Parent directories are created.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
        handler=handler,
    )


def edit() -> Tool:
    """Exact-string file editor: old_text must match exactly once."""

    def handler(args: dict[str, Any], ctx: Any) -> str:
        path = ctx.resolve(args["path"], must_exist=True)
        old = args["old_text"]
        new = args["new_text"]
        if not old:
            raise ValueError("old_text must be non-empty")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise ValueError(
                "old_text not found in file (it must match exactly, including whitespace)"
            )
        if count > 1:
            raise ValueError(
                f"old_text matches {count} times; include more context to make it unique"
            )
        path.write_text(text.replace(old, new), encoding="utf-8")
        return f"Edited {path}: replaced 1 occurrence"

    return Tool(
        name="edit",
        description=(
            "Replace an exact text snippet in a file. old_text must appear exactly once; "
            "include enough surrounding context to make it unique."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (must be unique in the file)",
                },
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        handler=handler,
    )
