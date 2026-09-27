"""Built-in shell tool: bash. Opt-in — gate it with an on_tool_call hook."""

from __future__ import annotations

import asyncio
from typing import Any

from aimee.tools.base import Tool


def bash() -> Tool:
    """Shell executor: runs in the primary workspace root."""

    async def handler(args: dict[str, Any], ctx: Any) -> str:
        command = str(args["command"])
        timeout = float(args.get("timeout") or ctx.config.bash_timeout)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(ctx.primary_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            await proc.wait()
            return f"[exit -1] Command timed out after {timeout:g}s"
        text = out.decode("utf-8", errors="replace")
        limit = ctx.config.output_limit
        if len(text) > limit:
            text = text[:limit] + f"\n[output truncated to {limit} chars]"
        return f"[exit {proc.returncode}]\n{text}"

    return Tool(
        name="bash",
        description=(
            "Run a shell command in the primary workspace directory. "
            "Returns combined stdout/stderr prefixed with the exit code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command line"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
            },
            "required": ["command"],
        },
        handler=handler,
    )
