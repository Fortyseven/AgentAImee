"""Built-in filesystem tools: read, write, edit. All opt-in."""

from __future__ import annotations

from typing import Any

from aimee.tools.base import Tool

DEFAULT_READ_LIMIT = 2000


def read() -> Tool:
    """File reader: numbered lines with offset/limit paging."""

    def handler(args: dict[str, Any], ctx: Any) -> str:
        path = ctx.resolve(args["path"], must_exist=True)
        if path.is_dir():
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in path.iterdir())
            return "\n".join(entries) if entries else "[empty directory]"
        offset = max(int(args.get("offset", 1)), 1)
        limit = max(int(args.get("limit", DEFAULT_READ_LIMIT)), 1)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        chunk = lines[offset - 1 : offset - 1 + limit]
        out = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
        if not chunk:
            out += f"\n[empty file, or offset {offset} is past end ({total} lines)]"
        elif offset - 1 + limit < total:
            shown_to = offset - 1 + len(chunk)
            out += f"\n[showing lines {offset}-{shown_to} of {total}; "
            out += f"pass offset={shown_to + 1} to continue]"
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
