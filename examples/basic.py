#!/usr/bin/env python3
"""Drop-in example: build an Aimee agent with basic tools, skills, and AGENTS.md.

Requires an OpenAI-compatible endpoint:

    export OPENAI_API_BASE=https://api.openai.com/v1   # or any compatible gateway
    export OPENAI_API_KEY=sk-...

Usage:
    uv run python examples/basic.py                      # runs a default demo task
    uv run python examples/basic.py "your task here"     # run your own task
    uv run python examples/basic.py --repl               # interactive REPL
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aimee import Aimee, AimeeConfig, RunReport
from aimee.tools import basic_tools

HERE = Path(__file__).resolve().parent

DEFAULT_TASK = (
    "Greet me (check your skills for how), read AGENTS.md and "
    "secondary/secret-note.txt, then summarize all three in three bullet points."
)


class Console:
    """Minimal hook set: print streamed tokens, log tool calls, summarize at the end."""

    def on_delta(self, chunk: str) -> None:
        print(chunk, end="", flush=True)

    def on_tool_call(self, name: str, args: dict) -> None:
        preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"\n[tool] {name}({preview[:120]})", file=sys.stderr)
        return None  # None = allow (no ToolDecision)

    def on_error(self, error: Exception) -> None:
        print(f"\n[error] {error}", file=sys.stderr)

    def on_done(self, report: RunReport) -> None:
        print(f"\n[done] turns={report.turns} tool_calls={report.tool_calls}", file=sys.stderr)


def build_agent(model: str | None = None) -> Aimee:
    config = AimeeConfig(
        roots=[HERE, HERE / "secondary"],  # primary + secondary workspace roots
        skills_dirs=[HERE / "skills"],
        model=model or os.environ.get("AIMEE_MODEL") or "default",
        # agents_md: left as None → nearest AGENTS.md found walking up from roots[0]
    )
    return Aimee(
        config,
        tools=basic_tools(["read", "write", "edit"]),  # bash deliberately left out
        hooks=[Console()],
    )


def run_task(agent: Aimee, task: str) -> None:
    agent.run(task)
    print()


def repl(agent: Aimee) -> None:
    print("Aimee example REPL — type a task; empty line or 'exit' quits.")
    while True:
        try:
            task = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task or task.lower() in {"exit", "quit"}:
            break
        run_task(agent, task)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aimee drop-in example")
    parser.add_argument("task", nargs="?", default=None, help="Task to run")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL")
    parser.add_argument(
        "--model", default=None, help="Model name (default: $AIMEE_MODEL or 'default')"
    )
    args = parser.parse_args()

    agent = build_agent(args.model)
    if args.repl:
        repl(agent)
    else:
        run_task(agent, args.task or DEFAULT_TASK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
