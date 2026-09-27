"""Aimee: a very minimal LLM agent loop with tools and skills."""

from aimee.agent import Aimee
from aimee.client import OpenAIClient
from aimee.config import AimeeConfig
from aimee.skills import Skill, load_skills, parse_frontmatter
from aimee.tools.base import Tool, ToolContext
from aimee.types import (
    AimeeError,
    ChatResponse,
    Message,
    RunReport,
    StreamDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolDecision,
)

__all__ = [
    "Aimee",
    "AimeeConfig",
    "AimeeError",
    "ChatResponse",
    "Message",
    "OpenAIClient",
    "RunReport",
    "Skill",
    "StreamDelta",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolContext",
    "ToolDecision",
    "load_skills",
    "parse_frontmatter",
]

__version__ = "0.1.0"
