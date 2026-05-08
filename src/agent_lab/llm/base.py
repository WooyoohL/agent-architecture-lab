from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from agent_lab.core.messages import Message


DecisionKind = Literal["tool_call", "final"]


@dataclass(frozen=True)
class AssistantDecision:
    # LLM 不直接执行工具，只返回 runtime 可以理解的结构化决策。
    kind: DecisionKind
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    content: str | None = None

    @classmethod
    def tool_call(cls, tool_name: str, args: dict[str, Any]) -> AssistantDecision:
        return cls(kind="tool_call", tool_name=tool_name, args=args)

    @classmethod
    def final(cls, content: str) -> AssistantDecision:
        return cls(kind="final", content=content)


class LLMClient(Protocol):
    def next(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantDecision:
        """Return the assistant's next structured decision."""
