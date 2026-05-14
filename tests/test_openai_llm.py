from types import SimpleNamespace

from agent_lab.core.messages import ToolObservationMessage, UserMessage
from agent_lab.llm.openai_llm import (
    OpenAILLMClient,
    response_to_decision,
    tools_to_openai_schema,
)


def test_tools_to_openai_schema_converts_registry_tool_info() -> None:
    tools = [
        {
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]

    assert tools_to_openai_schema(tools) == [
        {
            "type": "function",
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]


def test_response_to_decision_converts_function_call() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="calculator",
                arguments='{"expression": "2 + 3"}',
            )
        ]
    )

    decision = response_to_decision(response)

    assert decision.kind == "tool_call"
    assert decision.tool_name == "calculator"
    assert decision.args == {"expression": "2 + 3"}


def test_response_to_decision_converts_text_output() -> None:
    response = SimpleNamespace(output=[], output_text="2 + 3 = 5")

    decision = response_to_decision(response)

    assert decision.kind == "final"
    assert decision.content == "2 + 3 = 5"


def test_openai_client_uses_injected_client_without_real_api_call() -> None:
    fake_client = _FakeOpenAIClient()
    llm = OpenAILLMClient(model="test-model", client=fake_client)

    decision = llm.next(
        messages=[
            UserMessage("calculate 2 + 3"),
            ToolObservationMessage("calculator", ok=True, output=5),
        ],
        tools=[
            {
                "name": "calculator",
                "description": "Evaluate arithmetic.",
                "input_schema": {"type": "object"},
            }
        ],
    )

    assert decision.kind == "final"
    assert decision.content == "2 + 3 = 5"
    assert fake_client.responses.last_request["model"] == "test-model"
    assert fake_client.responses.last_request["tools"] == [
        {
            "type": "function",
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {"type": "object"},
        }
    ]
    assert "[TOOL_OBSERVATION] calculator ok=True output=5" in (
        fake_client.responses.last_request["input"]
    )


class _FakeResponses:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(output=[], output_text="2 + 3 = 5")


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()
