from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UserMessage:
    # 用户目标进入 agent loop 的第一条消息。
    content: str
    timestamp: str = field(default_factory=utc_timestamp)
    type: str = field(default="user", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class AssistantMessage:
    # Assistant 的普通文本消息，不包含工具调用。
    content: str
    timestamp: str = field(default_factory=utc_timestamp)
    type: str = field(default="assistant", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ToolCallMessage:
    # LLM 只提出要调用哪个工具和参数；真正执行由 runtime 完成。
    tool_name: str
    args: dict[str, Any]
    content: str = ""
    timestamp: str = field(default_factory=utc_timestamp)
    type: str = field(default="assistant_tool_call", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "args": self.args,
        }


@dataclass(frozen=True)
class ToolObservationMessage:
    # 工具执行结果回到消息流，成为 LLM 下一步可观察的事实。
    tool_name: str
    ok: bool
    output: Any | None = None
    error: str | None = None
    content: str = ""
    timestamp: str = field(default_factory=utc_timestamp)
    type: str = field(default="tool_observation", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }


@dataclass(frozen=True)
class FinalMessage:
    # FinalMessage 标记 agent loop 已经结束。
    content: str
    timestamp: str = field(default_factory=utc_timestamp)
    type: str = field(default="final", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


Message = (
    UserMessage
    | AssistantMessage
    | ToolCallMessage
    | ToolObservationMessage
    | FinalMessage
)
