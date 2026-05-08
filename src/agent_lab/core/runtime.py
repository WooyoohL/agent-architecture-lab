from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

from agent_lab.core.messages import (
    FinalMessage,
    Message,
    ToolCallMessage,
    ToolObservationMessage,
    UserMessage,
)
from agent_lab.core.tools import Tool, ToolRegistry
from agent_lab.core.trace import TraceRecorder
from agent_lab.llm.base import AssistantDecision, LLMClient


@dataclass(frozen=True)
class AgentRunResult:
    ok: bool
    output: str | None = None
    error: str | None = None


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        max_steps: int,
        trace_recorder: TraceRecorder,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.trace_recorder = trace_recorder
        self.messages: list[Message] = []

    def run(self, user_input: str) -> AgentRunResult:
        user_message = UserMessage(user_input)
        self.messages = [user_message]
        self.trace_recorder.record(user_message)

        for _ in range(self.max_steps):
            decision = self.llm.next(self.messages, self.tool_registry.list_tools())
            if decision.kind == "final":
                return self._record_final(decision)

            if decision.kind == "tool_call":
                self._execute_tool_call(decision)
                continue

            return AgentRunResult(
                ok=False,
                error=f"Unsupported assistant decision: {decision.kind}",
            )

        return AgentRunResult(
            ok=False,
            error=f"Reached max_steps={self.max_steps} without final answer.",
        )

    def _record_final(self, decision: AssistantDecision) -> AgentRunResult:
        content = decision.content or ""
        final_message = FinalMessage(content)
        self.messages.append(final_message)
        self.trace_recorder.record(final_message)
        return AgentRunResult(ok=True, output=content)

    def _execute_tool_call(self, decision: AssistantDecision) -> None:
        tool_name = decision.tool_name or ""
        args = decision.args or {}

        tool_call_message = ToolCallMessage(tool_name, args)
        self.messages.append(tool_call_message)
        self.trace_recorder.record(tool_call_message)

        result = self.tool_registry.call(tool_name, args)
        observation = ToolObservationMessage(
            tool_name=tool_name,
            ok=result.ok,
            output=result.output,
            error=result.error,
        )
        self.messages.append(observation)
        self.trace_recorder.record(observation)


def create_calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Evaluate a simple arithmetic expression.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        fn=_calculator,
    )


def _calculator(args: dict[str, Any]) -> int | float:
    expression = args["expression"]
    parsed = ast.parse(expression, mode="eval")
    return _eval_arithmetic(parsed)


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("Only numeric constants are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operator_fn = _BIN_OPS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("Only +, -, *, and / are allowed.")
        return operator_fn(_eval_arithmetic(node.left), _eval_arithmetic(node.right))

    if isinstance(node, ast.UnaryOp):
        operator_fn = _UNARY_OPS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("Only unary + and - are allowed.")
        return operator_fn(_eval_arithmetic(node.operand))

    raise ValueError("Only simple arithmetic expressions are allowed.")
