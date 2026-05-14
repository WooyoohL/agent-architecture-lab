from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_lab.core.runtime import AgentRuntime, create_calculator_tool
from agent_lab.core.tools import ToolRegistry
from agent_lab.core.trace import TraceRecorder
from agent_lab.llm.openai_llm import OpenAILLMClient


def main() -> None:
    registry = ToolRegistry()
    registry.register(create_calculator_tool())

    trace = TraceRecorder()
    runtime = AgentRuntime(
        llm=OpenAILLMClient(),
        tool_registry=registry,
        max_steps=5,
        trace_recorder=trace,
    )

    result = runtime.run("calculate 2 + 3")

    print(trace.as_text())
    print(f"final_ok={result.ok} output={result.output} error={result.error}")


if __name__ == "__main__":
    main()
