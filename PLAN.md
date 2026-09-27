# Aimee — Minimal LLM Agent Loop Library (Python)

## Context

New Python library **Aimee**: a very minimal LLM agent loop supporting tools and skills, designed to be dropped into *any* existing Python project. The client owns configuration and tool management; Aimee provides the loop, the OpenAI-compatible client, skills, AGENTS.md support, and callback hooks for extension.

### Decisions (confirmed with user)

- **Deps**: one runtime dep — `httpx`. OpenAI integration implemented by hand (chat completions + SSE streaming), no `openai` package.
- **API shape**: sync-first public API over an async core. `Aimee.run(task)` works in any script (runs the async core on a dedicated background event-loop thread, so it also works inside an already-running loop). `await aimee.run_async(task)` for async callers. Tools and hooks may be sync *or* async callables.
- **Streaming**: token-level `on_delta` hook (SSE deltas) *plus* turn-level complete messages.
- **Skills**: `SKILL.md` dirs (YAML frontmatter `name`/`description` + markdown body). Aimee scans configurable skill dirs, injects a **catalog** (name — description — path) into the system prompt, agent reads the full file on demand via `read`. Frontmatter parsed by hand (2-line `key: value` parser) — no PyYAML dep.
- **Python**: 3.10+.
- **Multi-root workspace**: `roots: list[Path]` (default `[Path.cwd()]`) — clients can add secondary locations like `~/.myapp`; per-root resolution rules defined in config section. (rev 2)
- **Model**: default `model="default"` (many OpenAI-compatible gateways route on that name). (rev 2)
- **Tool selection**: built-in tools are individually importable factories (`read()`, `write()`, `edit()`, `bash()`) plus `basic_tools(names=None)` for subsets. (rev 2)

## Approach

### Package layout (`src` layout, uv-managed, hatchling)

```
pyproject.toml            # deps: httpx; dev: pytest, ruff
README.md
src/aimee/
  __init__.py             # exports: Aimee, AimeeConfig, Tool, ToolContext, RunReport, ToolDecision, Skill, load_skills
  types.py                # Message, ChatResponse, TokenUsage, RunReport, ToolDecision
  config.py               # AimeeConfig (dataclass)
  client.py               # OpenAIClient (httpx): chat() + stream chat, SSE parsing, auth header
  agent.py                # async agent loop + Aimee sync/async facade
  hooks.py                # hook protocols + sync/async dispatch helper
  skills.py               # SKILL.md discovery + frontmatter parse + catalog rendering
  prompt.py               # system prompt assembly: base + AGENTS.md + skills catalog
  tools/
    __init__.py           # read(), write(), edit(), bash() factories + basic_tools(names=None) (all opt-in)
    base.py               # Tool dataclass (name, description, parameters JSON-schema, handler) + ToolContext
    fs.py                 # read, write, edit
    bash.py               # bash
examples/
  basic.py                # drop-in example: config, basic tools, skills, AGENTS.md, prints deltas, mini REPL
  custom_tool.py          # custom tool registration + tool-approval hook demo
tests/
  test_client.py
  test_agent.py
  test_tools.py
  test_skills.py
  test_prompt.py
  test_hooks.py
```

### Core design

**`AimeeConfig`** (dataclass, all optional with sane defaults):
- `roots: list[Path] = [Path.cwd()]` — one or more workspace roots. `roots[0]` is the primary root; extra roots (e.g. `~/.myapp`) extend where the agent may operate. Resolution semantics: `read`/`edit` try each root in order and use the first where the file **exists**; `write` uses the first root where the file exists, else `roots[0]`; `bash` runs with `cwd=roots[0]`; AGENTS.md walk-up tries each root in order (first found wins)
- `model: str = "default"`
- `api_base: str | None` — defaults to env `OPENAI_API_BASE`
- `api_key: str | None` — defaults to env `OPENAI_API_KEY`
- `system_prompt: str | None` — client-provided base prompt (replaces built-in minimal one)
- `agents_md: Path | None = None` — explicit AGENTS.md; else walk up from each `roots` entry in order (first found wins, stop at filesystem root)
- `skills_dirs: list[Path] = []`
- `max_turns: int = 30`
- `temperature: float | None = None`, `max_tokens: int | None = None`
- `bash_timeout: int = 120`, `output_limit: int = 100_000` (tool output truncation)

**`OpenAIClient`** (`client.py`, httpx):
- `POST {api_base}/chat/completions` with `Authorization: Bearer {key}`, model, messages, tools (OpenAI function-calling schema).
- `chat(...)` → full `ChatResponse`; `stream_chat(...)` → iterator of SSE deltas (hand-parsed: `data:` lines, `[DONE]`, content/tool_call arg deltas accumulated per index).
- HTTP errors → `AimeeError` with status + body snippet. No retries (keep minimal; client can wrap).

**`Aimee` facade** (`agent.py`):
- `__init__(config=None, *, client=None, tools=(), hooks=())` — `client` injectable (test seam).
- `add_tool(Tool)`, `add_hook(obj)`; `run(task) -> RunReport` (sync), `async run_async(task) -> RunReport`.
- Loop: build system prompt once (base + AGENTS.md + skills catalog) → append user message → call model with tools → if `tool_calls`: dispatch each (approval hook → execute → `role:"tool"` result message) → repeat until no tool calls or `max_turns` (→ report `truncated=True`).
- Tool execution errors are returned to the model as `Error: ...` content (so it can recover) **and** fired to `on_error`.
- `RunReport`: `final_text`, `turns`, `tool_calls` count, `usage` (accumulated `TokenUsage` when provider supplies it), `truncated`, `messages` (full transcript).

**Hooks** (`hooks.py`) — duck-typed; register any object exposing any subset of:
- `on_delta(chunk: str)` — streaming tokens
- `on_tool_call(name, args: dict) -> ToolDecision | None` — approval gate: `None`/`allow()` = run; `ToolDecision.deny(reason)` = skip, reason goes to model; `ToolDecision.modify(new_args)` = run with replaced args
- `on_tool_result(name, args, result: str)`
- `on_turn(turn: int, response: ChatResponse)`
- `on_error(error: Exception)`
- `on_done(report: RunReport)`
- All may be sync or async; dispatch helper handles both.

**Built-in tools** (all opt-in; each tool is a parameterless factory, and clients can pick exactly what they want):
- `from aimee.tools import read, write, edit, bash, basic_tools`
- `Aimee(tools=[read()])` — just `read`; `Aimee(tools=basic_tools())` — all four; `Aimee(tools=basic_tools(["read", "edit"]))` — named subset.
- Handler signature is `handler(args: dict, ctx: ToolContext) -> str`; Aimee builds `ToolContext` (roots, output_limit, bash_timeout, config) per call, so built-ins and client custom tools share the same workspace view.
- `read(path, offset=1, limit=2000)` — numbered lines, relative paths resolve per the multi-root rule above, truncation notice when capped.
- `write(path, content)` — create/overwrite, mkdir parents.
- `edit(path, old_text, new_text)` — exact string replace; `old_text` must match **exactly once** (error message lists match count otherwise).
- `bash(command, timeout?)` — run in `roots[0]`, shell, combined stdout+stderr truncated to `output_limit`, returns `[exit N]\n<output>`. No built-in approval — the `on_tool_call` hook is the safety mechanism (documented).

**System prompt** (`prompt.py`): built-in minimal base (identity + "use tools when needed") or `config.system_prompt`, then `# AGENTS.md` section (explicit path, else first found walking up from `roots` in order) under a heading, then `# Skills` catalog block:
```
- name — description
  path/to/SKILL.md (read before use)
```

### Tests (pytest)

- `test_client.py` — `httpx.MockTransport`: request shape, auth header, base URL join; `ChatResponse` parse incl. tool_calls; SSE stream delta accumulation (content + tool arg deltas); HTTP error → `AimeeError`.
- `test_agent.py` — scripted fake model (inject via `client=`): no-tools → done in 1 turn; tool call → result → second turn → done; denied tool call returns reason to model; `max_turns` truncation; usage accumulation; `run()` (sync) and `run_async()` parity.
- `test_tools.py` — `tmp_path`: read (numbering, offset/limit, missing file), write (parents created, overwrite), edit (unique match, no-match, multi-match error), bash (echo, nonzero exit, cwd, timeout), multi-root resolution (read/edit across `roots`, write defaulting to `roots[0]`).
- `test_skills.py` — frontmatter parse (valid, missing description, no frontmatter), discovery across multiple dirs, catalog rendering.
- `test_prompt.py` — AGENTS.md explicit path, walk-up discovery (tmp tree), nearest-wins, skills block present/absent, custom system_prompt replacement.
- `test_hooks.py` — sync + async hooks all dispatched; deny/modify decisions.

### Examples

- `examples/basic.py` — reads `OPENAI_API_BASE`/`KEY` from env, builds `AimeeConfig` (`roots=[Path.cwd(), Path.home()/".myapp"]`-style multi-root config, skills dir pointing at a `skills/` folder next to the example containing one tiny SKILL.md, `AGENTS.md` next to it), registers `basic_tools(["read", "write", "edit"])`, an `on_delta` hook that prints tokens, runs one task (or `--repl` for a tiny stdin loop), prints `RunReport`.
- `examples/custom_tool.py` — registers a single custom tool via `Tool(...)` using `ToolContext`, plus an `on_tool_call` hook that denies `bash` calls containing a marker string; prints report.

## Files to modify

All new (empty repo): tree above. No git repo exists yet — initialize `git init` + `.gitignore` (uv: `.venv`, `__pycache__`, `dist/`) as step 1.

## Steps

- [x] 1. `git init`, `.gitignore`, `pyproject.toml` (uv, hatchling, `httpx>=0.27`, pytest+ruff dev group), `uv sync`
- [x] 2. `types.py` + `config.py`
- [x] 3. `client.py` (OpenAIClient, SSE parsing) + `test_client.py`
- [x] 4. `hooks.py` + `prompt.py` + `agent.py` (loop + facade) + `test_agent.py`, `test_prompt.py`, `test_hooks.py`
- [x] 5. `skills.py` + `test_skills.py`
- [x] 6. `tools/` (base, fs, bash) + `test_tools.py`
- [x] 7. `examples/basic.py` + `examples/custom_tool.py` (+ their `AGENTS.md`/skill fixtures)
- [x] 8. `README.md` (quickstart, config/hook/tool/skill reference, security note on bash)
- [x] 9. `uv run ruff check` + `ruff format`, full pytest green

## Verification

- `uv run pytest -q` — all green.
- `uv run ruff check .` + `uv run ruff format --check .` clean.
- Manual (needs real key): `OPENAI_API_BASE=... OPENAI_API_KEY=... uv run python examples/basic.py "List the skills you have and read AGENTS.md, then summarize both."` — expect skill catalog + AGENTS.md content reflected in answer; `--repl` for interactive.
- `uv build` produces wheel + sdist without error (drop-in readiness check).
