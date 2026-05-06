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

Verified:

```bash
pytest tests/test_tools.py -q
python examples/00_tool_registry.py
```

Next:

- 1.2 message and trace infrastructure.
