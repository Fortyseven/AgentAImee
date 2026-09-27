#!/usr/bin/env python3
"""Example: client-defined tools + an on_tool_call approval hook.

Shows the two main extension points:
- a custom tool (`current_time`) built on the ToolContext, and
- a safety hook that denies `bash` calls containing a forbidden marker.

Requires OPENAI_API_BASE / OPENAI_API_KEY, like basic.py.

    uv run python examples/custom_tool.py
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

from aimee import Aimee, AimeeConfig, AimeeError, RunReport, Tool, ToolDecision
from aimee.tools import basic_tools

HERE = Path(__file__).resolve().parent

FORBIDDEN_MARKER = "rm "


def current_time(args: dict, ctx) -> str:
    """Custom tool handler: return the local date/time (ISO format)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


class SafetyHooks:
    """Approvals + console output in one duck-typed hook object."""

    def on_tool_call(self, name: str, args: dict) -> ToolDecision | None:
        if name == "bash" and FORBIDDEN_MARKER in str(args.get("command", "")):
            return ToolDecision.deny(
                f"commands containing {FORBIDDEN_MARKER!r} are not allowed here"
            )
        print(f"[tool] {name}", file=sys.stderr)
        return None

    def on_delta(self, chunk: str) -> None:
        print(chunk, end="", flush=True)

    def on_done(self, report: RunReport) -> None:
        print(f"\n[done] turns={report.turns} tool_calls={report.tool_calls}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aimee custom tool + approval hook example")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: $AIMEE_MODEL or the AimeeConfig default)",
    )
    args = parser.parse_args()
    # Precedence: --model flag → $AIMEE_MODEL → AimeeConfig default (omitted when unset).
    chosen = args.model or os.environ.get("AIMEE_MODEL")
    config = AimeeConfig(
        roots=[HERE, HERE / "secondary"],
        skills_dirs=[HERE / "skills"],
        **({"model": chosen} if chosen else {}),
    )
    agent = Aimee(
        config,
        tools=[
            *basic_tools(),  # all four built-ins, including bash (gated by the hook)
            Tool(
                name="current_time",
                description="Get the current local date and time (ISO format).",
                parameters={"type": "object", "properties": {}},
                handler=current_time,
            ),
        ],
        hooks=[SafetyHooks()],
    )
    task = (
        "Use current_time to say what time it is. Then read AGENTS.md and tell me "
        "the workspace safety policy. Finally try to run `bash rm --help` and report "
        "what happened."
    )
    try:
        agent.run(task)
    except AimeeError:
        print("\n[api error] see above — fix the endpoint/model and try again.", file=sys.stderr)
        return 1
    finally:
        agent.close()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
