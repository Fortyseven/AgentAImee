"""Core data types shared across Aimee."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


class AimeeError(Exception):
    """Error raised for failed OpenAI-compatible API calls."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class Message:
    """A single chat message in OpenAI wire format (plus light structure)."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        """Serialize to the dict shape OpenAI chat completions expects."""
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=ROLE_SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=ROLE_USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[dict[str, Any]] | None = None) -> Message:
        return cls(role=ROLE_ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> Message:
        return cls(role=ROLE_TOOL, content=content, tool_call_id=tool_call_id)


@dataclass
class TokenUsage:
    """Token accounting accumulated across turns (zero when the provider omits usage)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def is_zero(self) -> bool:
        return self.total_tokens == 0

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    @classmethod
    def from_openai(cls, data: dict[str, Any] | None) -> TokenUsage | None:
        """Build from an OpenAI `usage` object, or None when absent."""
        if not data:
            return None
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
        )


@dataclass
class ToolCall:
    """One tool invocation requested by the model (arguments already parsed)."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ToolCallDelta:
    """One SSE chunk's contribution to a tool call (arguments are a partial JSON string)."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class StreamDelta:
    """One SSE chunk from a streaming chat completion.

    `content` is empty when the chunk carried no text. Usage typically arrives
    in a final chunk with no content.
    """

    content: str = ""
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass
class ChatResponse:
    """One completed model response (full message or assembled from SSE deltas)."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolDecision:
    """Outcome of the `on_tool_call` approval hook for a single tool call."""

    allowed: bool = True
    args: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def allow(cls) -> ToolDecision:
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> ToolDecision:
        return cls(allowed=False, reason=reason)

    @classmethod
    def modify(cls, args: dict[str, Any]) -> ToolDecision:
        return cls(allowed=True, args=args)


@dataclass
class RunReport:
    """Summary of a finished agent run."""

    final_text: str
    turns: int
    tool_calls: int
    usage: TokenUsage = field(default_factory=TokenUsage)
    truncated: bool = False
    messages: list[Message] = field(default_factory=list)
