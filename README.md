# Aimee

A very minimal LLM agent loop for Python, with tools and skills. One runtime
dependency (`httpx`). Drop it into an existing project; you own the config,
the tools, and the behavior. Aimee gives you the loop, the OpenAI-compatible
client, skill discovery, AGENTS.md support, and callback hooks.

- **Model**: any OpenAI-compatible `chat/completions` endpoint via
  `OPENAI_API_BASE` + `OPENAI_API_KEY` (no `openai` package).
- **Tools**: built-in `read`, `write`, `edit`, `bash` — **all opt-in** — plus
  your own tools.
- **Skills**: `SKILL.md` directories; a catalog is injected into the system
  prompt and the agent reads the full skill on demand.
- **Hooks**: extend the run with callbacks — stream tokens, approve/modify
  tool calls, observe turns, handle errors.
- **Sync or async**: `agent.run(task)` works in any script; `await
  agent.run_async(task)` for async code.

## Setup

```bash
uv sync
uv run pytest -q          # test suite (no network needed)
```

## Quickstart

```bash
export OPENAI_API_BASE=https://api.openai.com/v1   # or any compatible gateway
export OPENAI_API_KEY=sk-...
uv run python examples/basic.py "your task here"   # or --repl
```

The example wires everything up: multi-root workspace, a skills directory,
AGENTS.md, three basic tools, and console hooks.

## Usage

```python
from pathlib import Path
from aimee import Aimee, AimeeConfig
from aimee.tools import basic_tools

config = AimeeConfig(
    roots=[Path.cwd(), Path.home() / ".myapp"],  # workspace roots (first = primary)
    skills_dirs=[Path.home() / ".myapp" / "skills"],
    model="gpt-4o-mini",  # default: "default"
)
agent = Aimee(config, tools=basic_tools())  # read, write, edit, bash
# agent = Aimee(config, tools=[read()])          # or just the tools you want

report = agent.run("Summarize AGENTS.md")  # sync (also safe inside a running loop)
# report = await agent.run_async("...")          # async

print(report.final_text, report.turns, report.tool_calls, report.usage)
```

`RunReport` fields: `final_text`, `turns`, `tool_calls`, `usage` (tokens,
when the provider reports them), `truncated` (max turns hit), `messages`
(full transcript).

## Configuration (`AimeeConfig`)

| Field | Default | Meaning |
| --- | --- | --- |
| `roots` | `[Path.cwd()]` | Workspace roots, in priority order (see below). All paths are configurable here. |
| `model` | `"default"` | Model name passed to the endpoint. |
| `api_base` | `$OPENAI_API_BASE` → `https://api.openai.com/v1` | Endpoint base URL. |
| `api_key` | `$OPENAI_API_KEY` | Bearer token. |
| `system_prompt` | built-in minimal prompt | Replaces the base prompt. |
| `agents_md` | nearest `AGENTS.md` walking up from `roots` | Explicit AGENTS.md path. |
| `skills_dirs` | `[]` | Directories containing `SKILL.md` skill folders. |
| `max_turns` | `30` | Model-call budget per run. |
| `temperature` / `max_tokens` | `None` | Passed through when set. |
| `bash_timeout` | `120` | Default shell timeout (seconds). |
| `output_limit` | `100_000` | Tool output truncation (chars). |

**Multi-root rules.** `roots[0]` is the primary root. `read`/`edit` resolve
relative paths against roots in order (first existing file wins); `write`
updates the first root that contains the file, and creates new files under
the primary root; `bash` runs with the primary root as cwd; AGENTS.md
discovery walks up from each root in order until one is found.

## Tools

Tools are OpenAI function-calling specs plus a handler:

```python
from aimee import Tool

Tool(
    name="current_time",
    description="Get the current local date and time (ISO format).",
    parameters={"type": "object", "properties": {}},  # JSON Schema
    handler=lambda args, ctx: "...",  # sync or async; returns str
)
agent.add_tool(tool)
```

`ctx` is a `ToolContext` giving handlers the same workspace view as the
built-ins: `ctx.roots`, `ctx.resolve(path, must_exist=...)`,
`ctx.config.bash_timeout`, `ctx.config.output_limit`.

Built-ins (opt-in): `basic_tools()` → all four, `basic_tools(["read", "edit"])`
→ a subset, or import individually: `from aimee.tools import read, write, edit, bash`.

| Tool | Behavior |
| --- | --- |
| `read(path, offset?, limit?)` | Numbered lines (paged), or directory listing. |
| `write(path, content)` | Create/overwrite; parent dirs created. |
| `edit(path, old_text, new_text)` | Exact-string replace; `old_text` must match exactly once. |
| `bash(command, timeout?)` | Shell in the primary root; `[exit N]` + combined output, truncated. |

> **Security note.** `bash` has no built-in approval. The `on_tool_call` hook
> is the safety mechanism — see `examples/custom_tool.py` for a hook that
> denies commands containing `rm `. Don't enable `bash` without one.

## Hooks

Register any object defining any subset of these methods (sync or async),
via `Aimee(..., hooks=[...])` or `agent.add_hook(obj)`:

| Hook | Purpose |
| --- | --- |
| `on_delta(chunk: str)` | One streamed chunk of assistant text (streaming on by default). |
| `on_tool_call(name, args) -> ToolDecision \| None` | Approval gate. `None` = allow; `ToolDecision.deny(reason)` blocks (reason goes to the model); `ToolDecision.modify(new_args)` replaces args. First non-None decision wins. |
| `on_tool_result(name, args, result)` | After a tool finished (including denials/errors). |
| `on_turn(turn, response)` | After each completed model response. |
| `on_error(error)` | Tool errors (run continues, error text goes to the model) and API errors (run aborts). |
| `on_done(report)` | Once, when the run finishes. |

## Skills

A skill is a directory with a `SKILL.md`:

```
myapp/skills/
└── pdf-extract/
    └── SKILL.md
```

```markdown
---
name: pdf-extract
description: Extract text from PDF files. Use when the task involves PDFs.
---
Full instructions here. The agent reads this file (via the read tool)
before using the skill.
```

Set `skills_dirs=[...]`; Aimee injects only the catalog (name, description,
path) into the system prompt, so skills cost almost nothing until used.
Frontmatter is parsed minimally (flat `key: value`, no PyYAML).

## AGENTS.md

If `agents_md` is unset, Aimee finds the nearest `AGENTS.md` walking up from
each root (in root order) and includes it under a "Project instructions"
heading. Give an explicit path to override.

## Tests

```bash
uv run pytest -q
```

The suite is fully offline: the HTTP client is tested against
`httpx.MockTransport`, and the agent loop against a scripted fake model.

## Project layout

```
src/aimee/
  agent.py    # the loop + Aimee facade (sync/async)
  client.py   # OpenAI-compatible client (chat + SSE streaming)
  config.py   # AimeeConfig
  hooks.py    # hook dispatch (sync/async)
  prompt.py   # system prompt assembly (base + AGENTS.md + skills)
  skills.py   # SKILL.md discovery + catalog
  types.py    # messages, responses, reports
  tools/      # base primitives + built-in read/write/edit/bash
examples/     # basic.py, custom_tool.py + workspace fixtures
tests/        # offline test suite
```
