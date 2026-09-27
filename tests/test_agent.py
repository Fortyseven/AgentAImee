"""Agent loop tests using a scripted fake model (no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from aimee import (
    Aimee,
    AimeeConfig,
    AimeeError,
    ChatResponse,
    StreamDelta,
    TokenUsage,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolDecision,
)
from aimee.tools import read

USAGE_1 = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
USAGE_2 = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)


def reply(text: str = "", calls=(), usage: TokenUsage | None = None) -> ChatResponse:
    tool_calls = [
        ToolCall(id=f"call_{i}", name=name, arguments=args, raw_arguments=json.dumps(args))
        for i, (name, args) in enumerate(calls)
    ]
    return ChatResponse(
        content=text,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class FakeModel:
    """Scripted stand-in for OpenAIClient: pops one ChatResponse per request."""

    def __init__(self, responses: list[ChatResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _next(self, messages, tools) -> ChatResponse:
        self.calls.append(
            {"messages": [m.to_openai() for m in messages], "tools": list(tools or [])}
        )
        if not self.responses:
            raise AssertionError("FakeModel ran out of scripted responses")
        return self.responses.pop(0)

    async def chat(self, messages, tools=None) -> ChatResponse:
        return self._next(messages, tools)

    async def stream_chat(self, messages, tools=None):
        resp = self._next(messages, tools)
        if resp.content:
            mid = max(len(resp.content) // 2, 0)
            for chunk in (resp.content[:mid], resp.content[mid:]):
                if chunk:
                    yield StreamDelta(content=chunk)
        for i, tc in enumerate(resp.tool_calls):
            yield StreamDelta(
                tool_calls=[
                    ToolCallDelta(index=i, id=tc.id, name=tc.name, arguments=tc.raw_arguments)
                ]
            )
        if resp.usage:
            yield StreamDelta(usage=resp.usage)
        yield StreamDelta(finish_reason=resp.finish_reason)


def spy_tool(name: str = "spy", result: str = "ok", record: list | None = None) -> Tool:
    def handler(args, ctx):
        if record is not None:
            record.append((name, args))
        return result

    return Tool(
        name=name,
        description="test spy",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_agent(tmp_path: Path, responses, tools=(), hooks=(), **cfg) -> tuple[Aimee, FakeModel]:
    config = AimeeConfig(roots=[tmp_path], api_key="sk-test", **cfg)
    fake = FakeModel(responses)
    agent = Aimee(config, client=fake, tools=tools, hooks=hooks)
    return agent, fake


def run(agent: Aimee, task: str = "do it"):
    return asyncio.run(agent.run_async(task))


class Recorder:
    def __init__(self):
        self.deltas: list[str] = []
        self.turns: list[int] = []
        self.tool_results: list[tuple] = []
        self.errors: list[Exception] = []
        self.dones: list = []

    def on_delta(self, chunk):
        self.deltas.append(chunk)

    def on_turn(self, turn, response):
        self.turns.append(turn)

    def on_tool_result(self, name, args, result):
        self.tool_results.append((name, args, result))

    def on_error(self, error):
        self.errors.append(error)

    def on_done(self, report):
        self.dones.append(report)


def test_no_tools_single_turn(tmp_path):
    recorder = Recorder()
    agent, fake = make_agent(tmp_path, [reply("all done")], hooks=[recorder])
    report = run(agent)

    assert report.final_text == "all done"
    assert report.turns == 1
    assert report.tool_calls == 0
    assert not report.truncated
    assert len(fake.calls) == 1
    # transcript: system + user
    assert [m.role for m in report.messages] == ["system", "user"]
    assert "Aimee" in report.messages[0].content
    assert recorder.dones == [report]
    assert recorder.turns == [1]


def test_tool_call_roundtrip_with_real_read_tool(tmp_path):
    (tmp_path / "notes.md").write_text("hello world", encoding="utf-8")
    recorder = Recorder()
    agent, fake = make_agent(
        tmp_path,
        [reply(calls=[("read", {"path": "notes.md"})]), reply("The note says hello world.")],
        tools=[read()],
        hooks=[recorder],
    )
    report = run(agent)

    assert report.final_text == "The note says hello world."
    assert report.turns == 2
    assert report.tool_calls == 1
    # the read result (numbered line) reached the model on turn 2
    second_call_messages = fake.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "hello world" in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == "call_0"
    # assistant message carried the tool call in wire format
    assistant = [m for m in second_call_messages if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "read"
    assert recorder.tool_results[0][0] == "read"


def test_denied_tool_not_executed(tmp_path):
    record: list = []

    class Denier:
        def on_tool_call(self, name, args):
            return ToolDecision.deny("not allowed")

    agent, _ = make_agent(
        tmp_path,
        [reply(calls=[("spy", {})]), reply("understood")],
        tools=[spy_tool(record=record)],
        hooks=[Denier()],
    )
    report = run(agent)

    assert record == []  # never executed
    tool_messages = [m for m in report.messages if m.role == "tool"]
    assert tool_messages[0].content == "Denied by client: not allowed"
    assert report.final_text == "understood"


def test_modified_args_used(tmp_path):
    record: list = []

    class Rewriter:
        def on_tool_call(self, name, args):
            return ToolDecision.modify({"v": 42})

    agent, _ = make_agent(
        tmp_path,
        [reply(calls=[("spy", {"v": 1})]), reply("ok")],
        tools=[spy_tool(record=record)],
        hooks=[Rewriter()],
    )
    report = run(agent)

    assert record == [("spy", {"v": 42})]
    assert report.final_text == "ok"


def test_first_decision_wins(tmp_path):
    class AllowAll:
        def on_tool_call(self, name, args):
            return ToolDecision.allow()

    class Denier:
        def on_tool_call(self, name, args):
            return ToolDecision.deny("late")

    record: list = []
    agent, _ = make_agent(
        tmp_path,
        [reply(calls=[("spy", {})]), reply("ok")],
        tools=[spy_tool(record=record)],
        hooks=[AllowAll(), Denier()],
    )
    run(agent)
    assert record == [("spy", {})]  # first hook (allow) decided


def test_max_turns_truncation(tmp_path):
    agent, fake = make_agent(
        tmp_path,
        [reply(calls=[("spy", {})]) for _ in range(5)],
        tools=[spy_tool()],
        max_turns=2,
    )
    report = run(agent)

    assert report.truncated
    assert report.turns == 2
    assert report.tool_calls == 2
    assert len(fake.calls) == 2
    assert "max turns" in report.final_text


def test_usage_accumulates(tmp_path):
    agent, _ = make_agent(
        tmp_path,
        [reply(calls=[("spy", {})], usage=USAGE_1), reply("done", usage=USAGE_2)],
        tools=[spy_tool()],
    )
    report = run(agent)
    assert report.usage.total_tokens == USAGE_1.total_tokens + USAGE_2.total_tokens
    assert report.usage.prompt_tokens == 13


def test_streaming_deltas_and_assembly(tmp_path):
    recorder = Recorder()
    agent, fake = make_agent(
        tmp_path,
        [reply("Hello world", calls=[("spy", {"a": 1})]), reply("done")],
        tools=[spy_tool()],
        hooks=[recorder],
    )
    agent.stream = True
    report = run(agent)

    assert recorder.deltas[:2] == ["Hello", " world"]  # turn 1, split by the fake
    assert "".join(recorder.deltas) == "Hello world" + "done"  # + turn 2 ("do"/"ne")
    assert report.final_text == "done"
    assert report.tool_calls == 1
    # streamed tool call assembled and parsed
    tool_messages = [m for m in report.messages if m.role == "tool"]
    assert "ok" in tool_messages[0].content
    assert fake.calls[0]["messages"][0]["role"] == "system"


def test_unknown_tool_reports_error(tmp_path):
    agent, _ = make_agent(tmp_path, [reply(calls=[("ghost", {})]), reply("ok")])
    report = run(agent)
    tool_messages = [m for m in report.messages if m.role == "tool"]
    assert tool_messages[0].content == "Error: unknown tool: 'ghost'"


def test_tool_exception_is_model_visible(tmp_path):
    def boom(args, ctx):
        raise ValueError("kaboom")

    recorder = Recorder()
    agent, _ = make_agent(
        tmp_path,
        [reply(calls=[("bad", {})]), reply("ok")],
        tools=[Tool(name="bad", description="d", parameters={}, handler=boom)],
        hooks=[recorder],
    )
    report = run(agent)

    tool_messages = [m for m in report.messages if m.role == "tool"]
    assert tool_messages[0].content == "Error: ValueError: kaboom"
    assert len(recorder.errors) == 1
    assert isinstance(recorder.errors[0], ValueError)


def test_api_error_propagates_and_fires_hook(tmp_path):
    class ExplodingModel:
        async def chat(self, messages, tools=None):
            raise AimeeError("boom upstream", status=500)

    recorder = Recorder()
    config = AimeeConfig(roots=[tmp_path], api_key="sk-test")
    agent = Aimee(config, client=ExplodingModel(), hooks=[recorder])
    agent.stream = False

    with pytest.raises(AimeeError, match="boom upstream"):
        run(agent)
    assert len(recorder.errors) == 1
    assert recorder.dones == []  # on_done not fired on abort


def test_sync_run_parity(tmp_path):
    (tmp_path / "x.txt").write_text("data", encoding="utf-8")
    fake = FakeModel([reply(calls=[("read", {"path": "x.txt"})]), reply("sync done")])
    config = AimeeConfig(roots=[tmp_path], api_key="sk-test")
    agent = Aimee(config, client=fake, tools=[read()])

    report = agent.run("read it")

    assert report.final_text == "sync done"
    assert report.turns == 2


def test_duplicate_tool_names_rejected(tmp_path):
    agent, _ = make_agent(tmp_path, [reply("hi")], tools=[spy_tool("dup"), spy_tool("dup")])
    with pytest.raises(ValueError, match="Duplicate tool name"):
        run(agent)


def test_system_prompt_has_agents_md_and_skills(tmp_path):
    (tmp_path / "AGENTS.md").write_text("be nice", encoding="utf-8")
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\nBody.", encoding="utf-8"
    )
    fake = FakeModel([reply("hi")])
    config = AimeeConfig(roots=[tmp_path], skills_dirs=[tmp_path / "skills"], api_key="sk-test")
    agent = Aimee(config, client=fake)
    run(agent)

    system = fake.calls[0]["messages"][0]["content"]
    assert "be nice" in system
    assert "demo" in system
    assert "A demo skill." in system
    assert str(skill_dir / "SKILL.md") in system
