"""Session tests: multi-turn conversations that keep chat history (no network)."""

from __future__ import annotations

from pathlib import Path

from aimee import Aimee, AimeeConfig, RunReport
from aimee.tools import read
from test_agent import FakeModel, reply, run


def make(tmp_path: Path, responses, tools=()) -> tuple[Aimee, FakeModel]:
    config = AimeeConfig(roots=[tmp_path], api_key="sk-test")
    fake = FakeModel(responses)
    agent = Aimee(config, client=fake, tools=tools)
    return agent, fake


def test_session_remembers_across_runs(tmp_path: Path):
    """The core feature: the second turn sees the first turn's conversation."""
    agent, fake = make(
        tmp_path, [reply("first answer"), reply("second answer, you said hello before")]
    )
    session = agent.session()

    r1 = session.run("say hello")
    assert isinstance(r1, RunReport)
    assert r1.final_text == "first answer"

    r2 = session.run("what did I just say?")
    assert r2.final_text == "second answer, you said hello before"

    # The second request must carry the whole prior conversation, system first.
    second_request = fake.calls[1]["messages"]
    flat = [(m["role"], m.get("content")) for m in second_request]
    assert second_request[0]["role"] == "system"
    assert ("user", "say hello") in flat
    assert ("assistant", "first answer") in flat
    assert ("user", "what did I just say?") in flat


def test_session_exposes_history(tmp_path: Path):
    agent, fake = make(tmp_path, [reply("one"), reply("two")])
    session = agent.session()

    assert session.history == []
    session.run("a")
    history = session.history
    assert [m.role for m in history] == ["system", "user", "assistant"]
    # report.messages mirrors the session history
    session.run("b")
    assert len(session.history) == 5


def test_session_clear_starts_over(tmp_path: Path):
    agent, fake = make(tmp_path, [reply("one"), reply("two")])
    session = agent.session()

    session.run("task-aaa")
    session.clear()
    assert session.history == []

    session.run("task-bbb")
    second_request = fake.calls[1]["messages"]
    contents = [m.get("content") for m in second_request]
    assert "task-aaa" not in contents
    assert "task-bbb" in contents
    assert second_request[0]["role"] == "system"  # fresh system prompt rebuilt


def test_sessions_on_one_agent_are_independent(tmp_path: Path):
    agent, fake = make(tmp_path, [reply("A1"), reply("B1"), reply("A2")])
    s_a, s_b = agent.session(), agent.session()

    s_a.run("alpha")
    s_b.run("beta")
    s_a.run("again")

    third_request = fake.calls[2]["messages"]
    contents = [m.get("content") for m in third_request]
    assert "alpha" in contents and "A1" in contents
    assert "beta" not in contents  # s_a never saw s_b's conversation


def test_session_keeps_tool_turns_in_history(tmp_path: Path):
    (tmp_path / "x.txt").write_text("data", encoding="utf-8")
    agent, fake = make(
        tmp_path,
        [
            reply(calls=[("read", {"path": "x.txt"})]),
            reply("it says data"),
            reply("yes, I read it earlier"),
        ],
        tools=[read()],
    )
    session = agent.session()

    session.run("read x.txt and tell me what it says")
    assert [m.role for m in session.history] == ["system", "user", "assistant", "tool", "assistant"]

    session.run("what did the file say?")
    second_request = fake.calls[2]["messages"]
    assert any(m["role"] == "tool" for m in second_request)
    assert any(m.get("content") == "it says data" for m in second_request)


def test_stateless_run_stays_stateless(tmp_path: Path):
    """agent.run() must remain stateless; only sessions keep history."""
    agent, fake = make(tmp_path, [reply("one"), reply("two")])

    agent.run("task-aaa")
    agent.run("task-bbb")

    second_request = fake.calls[1]["messages"]
    contents = [m.get("content") for m in second_request]
    assert "task-aaa" not in contents
    assert "task-bbb" in contents


def test_session_async_parity(tmp_path: Path):
    agent, fake = make(tmp_path, [reply("one"), reply("two")])
    session = agent.session()

    assert run(session, "a").final_text == "one"  # the run() helper wraps asyncio.run
    assert run(session, "b").final_text == "two"
    assert [m.role for m in session.history] == ["system", "user", "assistant", "user", "assistant"]
