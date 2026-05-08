from __future__ import annotations

import json
from typing import Any

from agent_lab.core.messages import (
    AssistantMessage,
    FinalMessage,
    Message,
    ToolCallMessage,
    ToolObservationMessage,
    UserMessage,
)


class TraceRecorder:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def record(self, message: Message) -> None:
        # Trace 是 append-only 时间线，便于复现 agent 每一步。
        self._messages.append(message)

    def as_text(self) -> str:
        return "\n".join(_format_message(message) for message in self._messages)

    def as_json(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self._messages]


def _format_message(message: Message) -> str:
    if isinstance(message, UserMessage):
        return f"[USER] {message.content}"

    if isinstance(message, AssistantMessage):
        return f"[ASSISTANT] {message.content}"

    if isinstance(message, ToolCallMessage):
        args = json.dumps(message.args, ensure_ascii=False)
        return f"[ASSISTANT_TOOL_CALL] {message.tool_name} {args}"

    if isinstance(message, ToolObservationMessage):
        output = _format_value(message.output)
        error = _format_value(message.error)
        if message.ok:
            return (
                f"[TOOL_OBSERVATION] {message.tool_name} "
                f"ok=True output={output}"
            )
        return (
            f"[TOOL_OBSERVATION] {message.tool_name} "
            f"ok=False error={error}"
        )

    if isinstance(message, FinalMessage):
        return f"[FINAL] {message.content}"

    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
