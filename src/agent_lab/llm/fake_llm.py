from __future__ import annotations

from typing import Any, Iterable, Sequence

from agent_lab.core.messages import Message
from agent_lab.llm.base import AssistantDecision


class FakeLLM:
    def __init__(self, decisions: Iterable[AssistantDecision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []

    def next(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantDecision:
        # 保存快照，测试可以确认 runtime 是否把 observation 放回消息流。
        self.calls.append({"messages": list(messages), "tools": list(tools)})

        if not self._decisions:
            raise RuntimeError("FakeLLM has no remaining decisions.")

        return self._decisions.pop(0)
