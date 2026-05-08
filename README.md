# Agent Architecture Lab

This repository is a small lab for learning agent architecture from first
principles.

The learning order follows:

```text
single-agent loop
-> tool registry
-> trace / state / max_steps
-> agentic RAG
-> verifier / evaluator
-> coding agent mini
-> planner-executor
-> router / handoff / multi-agent
-> framework refactor
```

## Current Progress

Current stage: Week 1, minimal agent loop.

Completed:

- 1.1 minimal tool registry
  - Implemented `Tool`, `ToolResult`, and `ToolRegistry`.
  - Added input schema validation for required fields and basic JSON types.
  - Added structured error handling for unknown tools and tool exceptions.
  - Added unit tests in `tests/test_tools.py`.
  - Added runnable trace-style example in `examples/00_tool_registry.py`.
- 1.2 message and trace infrastructure
  - Implemented `UserMessage`, `AssistantMessage`, `ToolCallMessage`,
    `ToolObservationMessage`, and `FinalMessage`.
  - Added `TraceRecorder` with human-readable text output and structured JSON
    output.
  - Added unit tests in `tests/test_runtime.py`.
  - Added runnable trace example in `examples/01_message_trace.py`.
- 1.3 FakeLLM and minimal runtime
  - Implemented `LLMClient`, `AssistantDecision`, and `FakeLLM`.
  - Added `AgentRuntime` for the minimal loop: user input, LLM decision, tool
    execution, observation, and final answer.
  - Added a safe AST-based `calculator` tool without `eval`.
  - Added unit tests in `tests/test_agent_loop.py`.
  - Updated `examples/01_minimal_agent.py` to run the full minimal loop.

Verified:

```bash
pytest tests/test_tools.py -q
pytest tests/test_runtime.py -q
pytest tests/test_agent_loop.py -q
python examples/00_tool_registry.py
python examples/01_message_trace.py
python examples/01_minimal_agent.py
```

Next:

- 1.4 real OpenAI LLM adapter.
