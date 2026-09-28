"""Tests for the OpenAI-compatible client, using httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
import json
import ssl

import httpx
import pytest

from aimee import AimeeConfig
from aimee.client import OpenAIClient, parse_sse_line
from aimee.types import AimeeError, Message


def run(coro):
    return asyncio.run(coro)


class FakeTool:
    """Duck-typed tool: all the client needs is to_openai()."""

    def to_openai(self):
        return {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }


def make_client(handler, **cfg) -> OpenAIClient:
    config = AimeeConfig(api_base="http://testserver", api_key="sk-test", **cfg)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://testserver")
    return OpenAIClient(config, client=http)


def chat_json(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


PLAIN_REPLY = {
    "choices": [
        {"message": {"role": "assistant", "content": "Hello there"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def test_request_shape_auth_and_model():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return chat_json(PLAIN_REPLY)

    client = make_client(handler, model="gpt-4o-mini", temperature=0.2, max_tokens=100)
    run(client.chat([Message.user("hi")]))

    assert captured["url"] == "http://testserver/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 100
    assert "tools" not in body
    assert "stream" not in body


def test_chat_response_content_and_usage():
    client = make_client(lambda r: chat_json(PLAIN_REPLY))
    resp = run(client.chat([Message.user("hi")]))
    assert resp.content == "Hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage is not None and resp.usage.total_tokens == 15


def test_chat_response_tool_calls_parsed():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path": "a.md"}'},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    client = make_client(lambda r: chat_json(payload))
    resp = run(client.chat([Message.user("go")], tools=[FakeTool()]))
    assert resp.has_tool_calls
    assert [tc.name for tc in resp.tool_calls] == ["read", "bash"]
    assert resp.tool_calls[0].arguments == {"path": "a.md"}
    assert resp.tool_calls[0].id == "c1"


def test_invalid_json_arguments_raise():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{nope"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    client = make_client(lambda r: chat_json(payload))
    with pytest.raises(AimeeError, match="invalid JSON"):
        run(client.chat([Message.user("go")]))


def test_http_error_raises_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = make_client(handler)
    with pytest.raises(AimeeError) as excinfo:
        run(client.chat([Message.user("hi")]))
    assert excinfo.value.status == 429
    assert "rate limited" in str(excinfo.value)


def sse_line(obj: dict) -> str:
    return "data: " + json.dumps(obj) + "\n"


SSE_STREAM = (
    sse_line({"choices": [{"delta": {"content": "Hel"}}]})
    + "\n"
    + sse_line({"choices": [{"delta": {"content": "lo"}}]})
    + ": keep-alive\n"
    + sse_line(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "read", "arguments": '{"pat'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    + sse_line(
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'h":"a.md"}'}}]}}
            ]
        }
    )
    + sse_line({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    + sse_line(
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}}
    )
    + "data: [DONE]\n"
)


def test_stream_chat_yields_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=SSE_STREAM.encode(), headers={"content-type": "text/event-stream"}
        )

    client = make_client(handler)

    async def collect():
        return [d async for d in client.stream_chat([Message.user("hi")])]

    deltas = run(collect())

    assert [d.content for d in deltas] == ["Hel", "lo", "", "", "", ""]
    assert deltas[0].finish_reason is None
    assert deltas[4].finish_reason == "tool_calls"
    # tool call split across two chunks; joined arguments parse to the full object
    assert deltas[2].tool_calls[0].name == "read"
    assert deltas[2].tool_calls[0].arguments == '{"pat'
    assert deltas[3].tool_calls[0].arguments == 'h":"a.md"}'
    assert json.loads(deltas[2].tool_calls[0].arguments + deltas[3].tool_calls[0].arguments) == {
        "path": "a.md"
    }
    # usage arrives in a final chunk without choices
    assert deltas[5].usage is not None and deltas[5].usage.total_tokens == 10


def test_stream_http_error_raises_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal boom")

    client = make_client(handler)

    async def drain():
        async for _ in client.stream_chat([Message.user("hi")]):
            pass

    with pytest.raises(AimeeError) as excinfo:
        run(drain())
    assert excinfo.value.status == 500
    assert "internal boom" in str(excinfo.value)


def test_stream_body_includes_stream_options():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=b"data: [DONE]\n", headers={"content-type": "text/event-stream"}
        )

    client = make_client(handler)

    async def drain():
        async for _ in client.stream_chat([Message.user("hi")]):
            pass

    run(drain())
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


def real_client(**cfg) -> OpenAIClient:
    """Client that builds its own httpx transport (so verify= is exercised)."""
    config = AimeeConfig(api_base="http://testserver", api_key="sk-test", **cfg)
    return OpenAIClient(config)


def ssl_verify_mode(client: OpenAIClient) -> int:
    """httpx 0.28 exposes no public `verify`; inspect the transport's SSL context."""
    return client._client._transport._pool._ssl_context.verify_mode


def test_default_client_verifies_tls():
    client = real_client()
    assert ssl_verify_mode(client) == ssl.CERT_REQUIRED
    run(client.aclose())


def test_verify_tls_false_disables_verification():
    client = real_client(verify_tls=False)
    assert ssl_verify_mode(client) == ssl.CERT_NONE
    run(client.aclose())


def test_insecure_ssl_env_has_no_effect(monkeypatch):
    """Regression guard: the old INSECURE_SSL=1 backdoor must stay dead."""
    monkeypatch.setenv("INSECURE_SSL", "1")
    client = real_client()
    assert ssl_verify_mode(client) == ssl.CERT_REQUIRED
    run(client.aclose())


def test_parse_sse_line_ignores_noise():
    assert parse_sse_line("") is None
    assert parse_sse_line("event: message") is None
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line("data: {broken json") is None
    delta = parse_sse_line('data: {"choices":[{"delta":{"content":"x"}}]}')
    assert delta is not None and delta.content == "x"
