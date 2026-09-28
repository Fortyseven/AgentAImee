"""OpenAI-compatible chat completions client, implemented by hand on httpx.

Speaks the standard `POST {base}/chat/completions` API with function calling
and SSE streaming. No `openai` package involved.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

import httpx

from aimee.config import AimeeConfig
from aimee.types import AimeeError, ChatResponse, StreamDelta, TokenUsage, ToolCall, ToolCallDelta

DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_BODY_SNIPPET = 500


class OpenAIClient:
    """Async client for one OpenAI-compatible endpoint.

    Pass a prebuilt `httpx.AsyncClient` (e.g. with a MockTransport) for tests;
    otherwise one is created from the config and closed with `aclose()`.
    """

    def __init__(
        self,
        config: AimeeConfig,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
    ):
        self.config = config
        self._owns_client = client is None
        if client is None:
            timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
            client = httpx.AsyncClient(
                base_url=config.resolved_api_base().rstrip("/"),
                timeout=timeout,
                verify=os.environ.get("INSECURE_SSL") != "1",
            )
        self._client = client
        # Sent per-request so injected clients (tests, proxies) also authenticate.
        self._auth_headers = {"Authorization": f"Bearer {config.resolved_api_key()}"}

    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if this object created it)."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenAIClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _body(
        self, messages: Sequence[Any], tools: Iterable[Any] | None, *, stream: bool
    ) -> dict[str, Any]:
        """Build the chat completions request body."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_openai() for m in messages],
        }
        tool_list = list(tools) if tools else []
        if tool_list:
            body["tools"] = [t.to_openai() for t in tool_list]
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            raise AimeeError(
                f"OpenAI API error {resp.status_code}: {resp.text[:_BODY_SNIPPET]}",
                status=resp.status_code,
                body=resp.text[:_BODY_SNIPPET],
            )

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        """Parse and JSON-decode a message's tool_calls; raises AimeeError on bad JSON."""
        out: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or ""
            try:
                args = json.loads(raw) if raw else {}
            except json.JSONDecodeError as e:
                raise AimeeError(f"Model returned invalid JSON tool arguments: {raw!r}") from e
            if not isinstance(args, dict):
                raise AimeeError(f"Model returned non-object tool arguments: {raw!r}")
            out.append(
                ToolCall(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=args,
                    raw_arguments=raw,
                )
            )
        return out

    async def chat(
        self, messages: Sequence[Any], tools: Iterable[Any] | None = None
    ) -> ChatResponse:
        """One non-streamed completion, returned as a full ChatResponse."""
        try:
            resp = await self._client.post(
                "/chat/completions",
                json=self._body(messages, tools, stream=False),
                headers=self._auth_headers,
            )
        except httpx.HTTPError as e:
            raise AimeeError(f"OpenAI API request failed: {e}") from e
        self._raise_for_status(resp)
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message),
            usage=TokenUsage.from_openai(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def stream_chat(
        self, messages: Sequence[Any], tools: Iterable[Any] | None = None
    ) -> AsyncIterator[StreamDelta]:
        """Streaming completion, yielding one StreamDelta per SSE chunk.

        Usage (if the provider sends it) arrives on a final delta chunk.
        """
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=self._body(messages, tools, stream=True),
                headers=self._auth_headers,
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()  # streaming body must be read before .text
                self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    delta = parse_sse_line(line)
                    if delta is not None:
                        yield delta
        except httpx.HTTPError as e:
            raise AimeeError(f"OpenAI API request failed: {e}") from e


def parse_sse_line(line: str) -> StreamDelta | None:
    """Parse one SSE line from a chat completions stream.

    Returns None for keep-alive comments, non-data lines, and [DONE].
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    delta = StreamDelta(usage=TokenUsage.from_openai(data.get("usage")))
    choices = data.get("choices") or []
    if not choices:
        return delta
    choice = choices[0]
    msg = choice.get("delta") or {}
    delta.content = msg.get("content") or ""
    delta.finish_reason = choice.get("finish_reason")
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        delta.tool_calls.append(
            ToolCallDelta(
                index=int(tc.get("index", 0)),
                id=tc.get("id"),
                name=fn.get("name"),
                arguments=fn.get("arguments") or "",
            )
        )
    return delta
