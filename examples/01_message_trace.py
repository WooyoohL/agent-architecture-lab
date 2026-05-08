from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_lab.core.messages import (
    FinalMessage,
    ToolCallMessage,
    ToolObservationMessage,
    UserMessage,
)
from agent_lab.core.trace import TraceRecorder


def main() -> None:
    trace = TraceRecorder()

    trace.record(UserMessage("calculate 2 + 3"))
    trace.record(ToolCallMessage("calculator", {"expression": "2 + 3"}))
    trace.record(ToolObservationMessage("calculator", ok=True, output=5))
    trace.record(FinalMessage("2 + 3 = 5"))

    print(trace.as_text())


if __name__ == "__main__":
    main()
