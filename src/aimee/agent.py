"""The agent loop and the Aimee facade (sync + async entry points)."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Iterable
from typing import Any

from aimee.client import OpenAIClient
from aimee.config import AimeeConfig
from aimee.hooks import call_hook, fire
from aimee.prompt import build_system_prompt
from aimee.skills import load_skills
from aimee.tools.base import Tool, ToolContext, validate_tools
from aimee.types import (
    AimeeError,
    ChatResponse,
    Message,
    RunReport,
    TokenUsage,
    ToolCall,
    ToolDecision,
)

_MAX_TURN_NOTE = "Stopped: max turns reached."


class Aimee:
    """A minimal LLM agent: model + tools + skills + callback hooks.

    Sync usage (works in any script, even inside a running event loop)::

        agent = Aimee(config, tools=basic_tools())
        report = agent.run("Summarize AGENTS.md")

    Async usage::

        report = await Aimee(config, tools=basic_tools()).run_async("...")

    `client` accepts anything with `chat()`/`stream_chat()` (dependency
    injection for tests); otherwise an OpenAIClient is created and closed
    automatically per run.
    """

    def __init__(
        self,
        config: AimeeConfig | None = None,
        *,
        client: Any = None,
        tools: Iterable[Tool] = (),
        hooks: Iterable[Any] = (),
        stream: bool = True,
    ):
        self.config = config or AimeeConfig()
        self._client = client
        self._owns_client = client is None
        self.tools: list[Tool] = list(tools)
        self.hooks: list[Any] = list(hooks)
        self.stream = stream

    # -- registration ------------------------------------------------------

    def add_tool(self, tool: Tool) -> Aimee:
        """Register one tool; returns self for chaining."""
        self.tools.append(tool)
        return self

    def add_hook(self, hook: Any) -> Aimee:
        """Register a hook object (any subset of the on_* methods); chainable."""
        self.hooks.append(hook)
        return self

    # -- entry points ------------------------------------------------------

    def run(self, task: str) -> RunReport:
        """Run the agent (sync). Executes the async loop on a background thread."""
        return run_coroutine_on_thread(self.run_async(task))

    async def run_async(self, task: str) -> RunReport:
        """Run the agent (async). Creates/closes its own model client if needed."""
        created = self._client is None
        if created:
            self._client = OpenAIClient(self.config)
        try:
            return await self._run(task)
        finally:
            if created and self._client is not None:
                await self._client.aclose()

    async def aclose(self) -> None:
        """Close a model client Aimee created itself (no-op for injected clients)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    # -- the loop ----------------------------------------------------------

    async def _run(self, task: str) -> RunReport:
        tools = validate_tools(self.tools)
        ctx = ToolContext(config=self.config, roots=self.config.resolved_roots())
        skills = load_skills(self.config.skills_dirs) if self.config.skills_dirs else []
        messages: list[Message] = [
            Message.system(build_system_prompt(self.config, skills)),
            Message.user(task),
        ]
        usage = TokenUsage()
        tool_call_count = 0
        turns = 0
        truncated = False
        final_text = ""

        try:
            while True:
                if turns >= self.config.max_turns:
                    truncated = True
                    break
                turns += 1
                response = await self._request_turn(messages, tools)
                if response.usage:
                    usage.add(response.usage)
                await fire(self.hooks, "on_turn", turns, response)

                if not response.has_tool_calls:
                    final_text = response.content
                    break

                messages.append(_assistant_message(response))
                for tool_call in response.tool_calls:
                    tool_call_count += 1
                    result, args_used = await self._dispatch_tool(tool_call, tools, ctx)
                    await fire(self.hooks, "on_tool_result", tool_call.name, args_used, result)
                    messages.append(Message.tool_result(tool_call.id, result))
        except AimeeError as e:
            await fire(self.hooks, "on_error", e)
            raise

        report = RunReport(
            final_text=final_text or (_MAX_TURN_NOTE if truncated else ""),
            turns=turns,
            tool_calls=tool_call_count,
            usage=usage,
            truncated=truncated,
            messages=messages,
        )
        await fire(self.hooks, "on_done", report)
        return report

    async def _request_turn(self, messages: list[Message], tools: dict[str, Tool]) -> ChatResponse:
        """One model request; streams deltas to on_delta hooks when enabled."""
        tool_list = list(tools.values()) if tools else None
        if not self.stream:
            return await self._client.chat(messages, tools=tool_list)

        content_parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        async for delta in self._client.stream_chat(messages, tools=tool_list):
            if delta.content:
                content_parts.append(delta.content)
                await fire(self.hooks, "on_delta", delta.content)
            for tc in delta.tool_calls:
                slot = slots.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.name:
                    slot["name"] = tc.name
                slot["args"] += tc.arguments
            if delta.finish_reason:
                finish_reason = delta.finish_reason
            if delta.usage:
                usage = usage or TokenUsage()
                usage.add(delta.usage)

        tool_calls = [_assemble_tool_call(idx, slot) for idx, slot in sorted(slots.items())]
        return ChatResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    async def _dispatch_tool(
        self, tool_call: ToolCall, tools: dict[str, Tool], ctx: ToolContext
    ) -> tuple[str, dict[str, Any]]:
        """Apply the approval hook, execute the tool, and report the outcome."""
        decision = await self._decide(tool_call.name, tool_call.arguments)
        if decision is not None and not decision.allowed:
            return f"Denied by client: {decision.reason or 'no reason given'}", tool_call.arguments
        args = (
            decision.args
            if decision is not None and decision.args is not None
            else tool_call.arguments
        )
        tool = tools.get(tool_call.name)
        if tool is None:
            result = f"Error: unknown tool: {tool_call.name!r}"
        else:
            result = await self._execute_tool(tool, args, ctx)
        return result, args

    async def _decide(self, name: str, args: dict[str, Any]) -> ToolDecision | None:
        """First non-None ToolDecision from the hooks wins; None means allow."""
        for hook in self.hooks:
            decision = await call_hook(hook, "on_tool_call", name, args)
            if decision is not None:
                return decision
        return None

    async def _execute_tool(self, tool: Tool, args: dict[str, Any], ctx: ToolContext) -> str:
        """Run a tool handler; errors become model-visible text + on_error."""
        try:
            result = tool.handler(args, ctx)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            await fire(self.hooks, "on_error", e)
            return f"Error: {type(e).__name__}: {e}"


def _assistant_message(response: ChatResponse) -> Message:
    """Assistant message in OpenAI wire form (content + raw tool calls)."""
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.raw_arguments or json.dumps(tc.arguments),
            },
        }
        for tc in response.tool_calls
    ]
    return Message(role="assistant", content=response.content or None, tool_calls=tool_calls)


def _assemble_tool_call(index: int, slot: dict[str, str]) -> ToolCall:
    """Join streamed argument fragments and parse the final JSON object."""
    raw = slot["args"]
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        raise AimeeError(f"Model returned invalid JSON tool arguments: {raw!r}") from e
    if not isinstance(args, dict):
        raise AimeeError(f"Model returned non-object tool arguments: {raw!r}")
    return ToolCall(id=slot["id"], name=slot["name"], arguments=args, raw_arguments=raw)


def run_coroutine_on_thread(coro: Any) -> Any:
    """Run a coroutine on a dedicated background event-loop thread.

    Works even when the caller already has a running event loop (Jupyter,
    async frameworks, Tk). The exception, if any, is re-raised here.
    """
    box: list[Any] = []

    def worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            box.append(loop.run_until_complete(coro))
        except BaseException as e:  # surfaced to the caller below
            box.append(e)
        finally:
            loop.close()

    thread = threading.Thread(target=worker, name="aimee-loop", daemon=True)
    thread.start()
    thread.join()
    outcome = box[0]
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome
