from __future__ import annotations

import json
import os
from typing import Any, Sequence

from agent_lab.core.messages import (
    AssistantMessage,
    FinalMessage,
    Message,
    ToolCallMessage,
    ToolObservationMessage,
    UserMessage,
)
from agent_lab.llm.base import AssistantDecision


DEFAULT_MODEL = "gpt-5.4-mini"


DEFAULT_INSTRUCTIONS = """You are the decision maker inside a minimal agent loop.
Return a function tool call when a tool is needed.
Return a normal final answer when the trace already contains enough information.
Do not claim that you executed a tool yourself; the runtime executes tools."""


class OpenAILLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.instructions = instructions

        if client is not None:
            self.client = client
            return

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set to use OpenAILLMClient.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is required for OpenAILLMClient. "
                "Install it with: pip install openai"
            ) from exc

        self.client = OpenAI(api_key=api_key)

    def next(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantDecision:
        response = self.client.responses.create(
            model=self.model,
            input=_messages_to_input(messages),
            tools=tools_to_openai_schema(tools),
            instructions=self.instructions,
        )
        return response_to_decision(response)


def tools_to_openai_schema(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        }
        for tool in tools
    ]


def response_to_decision(response: Any) -> AssistantDecision:
    for item in _get_output_items(response):
        if _get_value(item, "type") != "function_call":
            continue

        name = _get_value(item, "name")
        arguments = _get_value(item, "arguments") or "{}"
        return AssistantDecision.tool_call(name, _parse_arguments(arguments))

    return AssistantDecision.final(_get_output_text(response))


def _messages_to_input(messages: Sequence[Message]) -> str:
    lines = ["Conversation trace:"]
    for message in messages:
        lines.append(_format_message_for_model(message))
    return "\n".join(lines)


def _format_message_for_model(message: Message) -> str:
    if isinstance(message, UserMessage):
        return f"[USER] {message.content}"

    if isinstance(message, AssistantMessage):
        return f"[ASSISTANT] {message.content}"

    if isinstance(message, ToolCallMessage):
        args = json.dumps(message.args, ensure_ascii=False)
        return f"[ASSISTANT_TOOL_CALL] {message.tool_name} {args}"

    if isinstance(message, ToolObservationMessage):
        if message.ok:
            output = json.dumps(message.output, ensure_ascii=False)
            return f"[TOOL_OBSERVATION] {message.tool_name} ok=True output={output}"
        return f"[TOOL_OBSERVATION] {message.tool_name} ok=False error={message.error}"

    if isinstance(message, FinalMessage):
        return f"[FINAL] {message.content}"

    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _get_output_items(response: Any) -> list[Any]:
    return list(_get_value(response, "output") or [])


def _get_output_text(response: Any) -> str:
    output_text = _get_value(response, "output_text")
    if output_text:
        return output_text

    text_parts: list[str] = []
    for item in _get_output_items(response):
        if _get_value(item, "type") != "message":
            continue
        for content_item in _get_value(item, "content") or []:
            if _get_value(content_item, "type") == "output_text":
                text = _get_value(content_item, "text")
                if text:
                    text_parts.append(text)
    return "\n".join(text_parts)


def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tool call JSON arguments: {arguments}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return parsed


def _get_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
