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

Current stage: Week 2, agentic RAG.

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
- 1.4 real OpenAI LLM adapter
  - Implemented `OpenAILLMClient` without changing `AgentRuntime`.
  - Converts local tool schemas to OpenAI function tool schemas.
  - Converts OpenAI function calls into `AssistantDecision.tool_call`.
  - Converts OpenAI text responses into `AssistantDecision.final`.
  - Added non-network adapter tests in `tests/test_openai_llm.py`.
  - Added real API example in `examples/01b_real_llm_agent.py`.
- 2.1 toy corpus and retriever tool
  - Implemented `Document` and `InMemoryCorpus`.
  - Added `SimpleKeywordRetriever` using keyword overlap.
  - Wrapped the retriever as `retrieve_docs` tool.
  - Returns evidence objects with `doc_id`, `title`, `snippet`, and `score`.
  - Added unit tests in `tests/test_agentic_rag.py`.
  - Updated `examples/02_agentic_rag.py` to run retrieval through the agent loop.
- 2.2 minimal agentic RAG loop
  - Updated `examples/02_agentic_rag.py` to show multi-step retrieval.
  - Uses `FakeLLM` to first issue a broad query, then rewrite to a narrower
    query after observing weak evidence.
  - Trace shows two `retrieve_docs` calls with different queries.
  - Final answer includes cited `doc_id` values.
  - Added tests for repeated retriever calls, cited final answers, and
    `max_steps` stopping behavior.
- 2.3 grounding verifier
  - Implemented `VerificationResult` and `SimpleGroundingVerifier`.
  - Added `verify_answer` tool to check citation presence and cited `doc_id`
    existence.
  - Updated `examples/02_agentic_rag.py` to show a missing-citation answer,
    verifier failure, repaired answer, and final grounded response.
  - Added tests for no citation, missing citation, valid citation, and verifier
    tool output.

Verified:

```bash
pytest tests/test_tools.py -q
pytest tests/test_runtime.py -q
pytest tests/test_agent_loop.py -q
pytest tests/test_openai_llm.py -q
pytest tests/test_agentic_rag.py -q
python examples/00_tool_registry.py
python examples/01_message_trace.py
python examples/01_minimal_agent.py
python examples/02_agentic_rag.py
```

To run the real OpenAI example:

```bash
pip install openai
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.4-mini"
python examples/01b_real_llm_agent.py
```

Next:

- 2.4 retry loop / verifier-driven answer repair.
