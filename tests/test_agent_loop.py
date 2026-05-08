from agent_lab.core.messages import ToolObservationMessage
from agent_lab.core.runtime import AgentRuntime, create_calculator_tool
from agent_lab.core.tools import ToolRegistry
from agent_lab.core.trace import TraceRecorder
from agent_lab.llm.base import AssistantDecision
from agent_lab.llm.fake_llm import FakeLLM


def test_agent_runtime_runs_tool_call_then_final() -> None:
    registry = ToolRegistry()
    registry.register(create_calculator_tool())
    trace = TraceRecorder()
    llm = FakeLLM(
        [
            AssistantDecision.tool_call("calculator", {"expression": "2 + 3"}),
            AssistantDecision.final("2 + 3 = 5"),
        ]
    )
    runtime = AgentRuntime(llm, registry, max_steps=3, trace_recorder=trace)

    result = runtime.run("calculate 2 + 3")

    assert result.ok is True
    assert result.output == "2 + 3 = 5"
    assert trace.as_text() == "\n".join(
        [
            "[USER] calculate 2 + 3",
            '[ASSISTANT_TOOL_CALL] calculator {"expression": "2 + 3"}',
            "[TOOL_OBSERVATION] calculator ok=True output=5",
            "[FINAL] 2 + 3 = 5",
        ]
    )

    second_llm_call_messages = llm.calls[1]["messages"]
    assert isinstance(second_llm_call_messages[-1], ToolObservationMessage)
    assert second_llm_call_messages[-1].output == 5


def test_agent_runtime_records_tool_error_and_continues() -> None:
    registry = ToolRegistry()
    registry.register(create_calculator_tool())
    trace = TraceRecorder()
    llm = FakeLLM(
        [
            AssistantDecision.tool_call("calculator", {"expression": "1 / 0"}),
            AssistantDecision.final("The calculator failed: division by zero."),
        ]
    )
    runtime = AgentRuntime(llm, registry, max_steps=3, trace_recorder=trace)

    result = runtime.run("calculate 1 / 0")

    assert result.ok is True
    assert result.output == "The calculator failed: division by zero."
    assert "[TOOL_OBSERVATION] calculator ok=False error=division by zero" in (
        trace.as_text()
    )

    second_llm_call_messages = llm.calls[1]["messages"]
    assert isinstance(second_llm_call_messages[-1], ToolObservationMessage)
    assert second_llm_call_messages[-1].ok is False
    assert second_llm_call_messages[-1].error == "division by zero"


def test_agent_runtime_stops_at_max_steps() -> None:
    registry = ToolRegistry()
    registry.register(create_calculator_tool())
    trace = TraceRecorder()
    llm = FakeLLM(
        [
            AssistantDecision.tool_call("calculator", {"expression": "2 + 3"}),
            AssistantDecision.final("This decision should not be reached."),
        ]
    )
    runtime = AgentRuntime(llm, registry, max_steps=1, trace_recorder=trace)

    result = runtime.run("calculate 2 + 3")

    assert result.ok is False
    assert result.output is None
    assert result.error == "Reached max_steps=1 without final answer."
    assert "[FINAL]" not in trace.as_text()
