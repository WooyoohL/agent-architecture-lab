from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_lab.core.runtime import AgentRuntime, create_calculator_tool
from agent_lab.core.tools import ToolRegistry
from agent_lab.core.trace import TraceRecorder
from agent_lab.llm.base import AssistantDecision
from agent_lab.llm.fake_llm import FakeLLM


def main() -> None:
    registry = ToolRegistry()
    registry.register(create_calculator_tool())

    llm = FakeLLM(
        [
            AssistantDecision.tool_call("calculator", {"expression": "2 + 3"}),
            AssistantDecision.final("2 + 3 = 5"),
        ]
    )
    trace = TraceRecorder()
    runtime = AgentRuntime(
        llm=llm,
        tool_registry=registry,
        max_steps=3,
        trace_recorder=trace,
    )

    result = runtime.run("calculate 2 + 3")

    print(trace.as_text())
    print(f"final_ok={result.ok} output={result.output} error={result.error}")


if __name__ == "__main__":
    main()
