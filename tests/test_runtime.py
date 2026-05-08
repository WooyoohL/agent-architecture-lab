from agent_lab.core.messages import (
    FinalMessage,
    ToolCallMessage,
    ToolObservationMessage,
    UserMessage,
)
from agent_lab.core.trace import TraceRecorder


def test_messages_serialize_to_dict() -> None:
    message = ToolCallMessage(
        tool_name="calculator",
        args={"expression": "2 + 3"},
        timestamp="2026-05-08T00:00:00+00:00",
    )

    assert message.to_dict() == {
        "type": "assistant_tool_call",
        "content": "",
        "timestamp": "2026-05-08T00:00:00+00:00",
        "tool_name": "calculator",
        "args": {"expression": "2 + 3"},
    }


def test_trace_as_text_formats_agent_steps() -> None:
    trace = TraceRecorder()
    trace.record(UserMessage("calculate 2 + 3"))
    trace.record(ToolCallMessage("calculator", {"expression": "2 + 3"}))
    trace.record(ToolObservationMessage("calculator", ok=True, output=5))
    trace.record(FinalMessage("2 + 3 = 5"))

    assert trace.as_text() == "\n".join(
        [
            "[USER] calculate 2 + 3",
            '[ASSISTANT_TOOL_CALL] calculator {"expression": "2 + 3"}',
            "[TOOL_OBSERVATION] calculator ok=True output=5",
            "[FINAL] 2 + 3 = 5",
        ]
    )


def test_trace_as_text_formats_tool_errors() -> None:
    trace = TraceRecorder()
    trace.record(ToolObservationMessage("calculator", ok=False, error="bad input"))

    assert trace.as_text() == "[TOOL_OBSERVATION] calculator ok=False error=bad input"


def test_trace_as_json_keeps_structured_messages() -> None:
    trace = TraceRecorder()
    trace.record(
        UserMessage(
            "calculate 2 + 3",
            timestamp="2026-05-08T00:00:00+00:00",
        )
    )
    trace.record(
        ToolObservationMessage(
            "calculator",
            ok=True,
            output=5,
            timestamp="2026-05-08T00:00:01+00:00",
        )
    )

    assert trace.as_json() == [
        {
            "type": "user",
            "content": "calculate 2 + 3",
            "timestamp": "2026-05-08T00:00:00+00:00",
        },
        {
            "type": "tool_observation",
            "content": "",
            "timestamp": "2026-05-08T00:00:01+00:00",
            "tool_name": "calculator",
            "ok": True,
            "output": 5,
            "error": None,
        },
    ]
