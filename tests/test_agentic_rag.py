from agent_lab.core.runtime import AgentRuntime
from agent_lab.core.tools import ToolRegistry
from agent_lab.core.trace import TraceRecorder
from agent_lab.llm.base import AssistantDecision
from agent_lab.llm.fake_llm import FakeLLM
from agent_lab.rag.corpus import Document, InMemoryCorpus
from agent_lab.rag.retriever import (
    Evidence,
    SimpleKeywordRetriever,
    create_retriever_tool,
)
from agent_lab.rag.verifier import SimpleGroundingVerifier, create_verifier_tool


def test_keyword_retriever_returns_matching_documents() -> None:
    retriever = SimpleKeywordRetriever(_build_corpus())

    results = retriever.retrieve("agent runtime", top_k=3)

    assert [result.doc_id for result in results] == ["agent-loop", "runtime"]
    assert results[0].score == 2
    assert "agent loop" in results[0].snippet.lower()


def test_keyword_retriever_respects_top_k() -> None:
    retriever = SimpleKeywordRetriever(_build_corpus())

    results = retriever.retrieve("agent tool runtime", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "agent-loop"


def test_keyword_retriever_returns_empty_list_for_no_match() -> None:
    retriever = SimpleKeywordRetriever(_build_corpus())

    assert retriever.retrieve("quantum banana", top_k=3) == []


def test_retriever_tool_schema_and_output() -> None:
    retriever = SimpleKeywordRetriever(_build_corpus())
    tool = create_retriever_tool(retriever)

    assert tool.schema_info() == {
        "name": "retrieve_docs",
        "description": "Retrieve relevant documents from the toy corpus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    }

    result = tool.call({"query": "trace", "top_k": 1})

    assert result.ok is True
    assert result.output == [
        {
            "doc_id": "trace",
            "title": "Trace Recorder",
            "snippet": (
                "Trace records user messages, tool calls, tool observations, "
                "and final answers for debugging."
            ),
            "score": 1,
        }
    ]


def test_agentic_rag_can_call_retriever_multiple_times() -> None:
    runtime, trace = _build_agentic_rag_runtime(
        [
            AssistantDecision.tool_call("retrieve_docs", {"query": "agent", "top_k": 1}),
            AssistantDecision.tool_call(
                "retrieve_docs",
                {"query": "tool observation returned to LLM context", "top_k": 1},
            ),
            AssistantDecision.final("Answer grounded in [agent-loop] and [observation]."),
        ],
        max_steps=4,
    )

    result = runtime.run("How do tool observations support agent runtime answers?")
    trace_text = trace.as_text()

    assert result.ok is True
    assert "[agent-loop]" in result.output
    assert "[observation]" in result.output
    assert '[ASSISTANT_TOOL_CALL] retrieve_docs {"query": "agent", "top_k": 1}' in (
        trace_text
    )
    assert (
        '[ASSISTANT_TOOL_CALL] retrieve_docs {"query": '
        '"tool observation returned to LLM context", "top_k": 1}'
    ) in trace_text
    assert trace_text.count("[ASSISTANT_TOOL_CALL] retrieve_docs") == 2


def test_agentic_rag_final_answer_contains_evidence_reference() -> None:
    runtime, _trace = _build_agentic_rag_runtime(
        [
            AssistantDecision.tool_call(
                "retrieve_docs",
                {"query": "runtime max steps", "top_k": 2},
            ),
            AssistantDecision.final("Runtime behavior is described in [runtime]."),
        ],
        max_steps=3,
    )

    result = runtime.run("What enforces max steps?")

    assert result.ok is True
    assert "[runtime]" in result.output


def test_agentic_rag_stops_at_max_steps() -> None:
    runtime, trace = _build_agentic_rag_runtime(
        [
            AssistantDecision.tool_call("retrieve_docs", {"query": "agent", "top_k": 1}),
            AssistantDecision.tool_call("retrieve_docs", {"query": "runtime", "top_k": 1}),
            AssistantDecision.final("This should not be reached."),
        ],
        max_steps=2,
    )

    result = runtime.run("Keep searching forever")

    assert result.ok is False
    assert result.error == "Reached max_steps=2 without final answer."
    assert trace.as_text().count("[ASSISTANT_TOOL_CALL] retrieve_docs") == 2
    assert "[FINAL]" not in trace.as_text()


def test_grounding_verifier_fails_without_citation() -> None:
    verifier = SimpleGroundingVerifier()

    result = verifier.verify(
        answer="The runtime turns tool results into observations.",
        evidence=_build_evidence(),
    )

    assert result.ok is False
    assert result.unsupported_claims == [
        "Answer does not include any evidence citation."
    ]
    assert result.missing_citations == []


def test_grounding_verifier_fails_for_missing_citation() -> None:
    verifier = SimpleGroundingVerifier()

    result = verifier.verify(
        answer="The runtime turns tool results into observations [missing-doc].",
        evidence=_build_evidence(),
    )

    assert result.ok is False
    assert result.unsupported_claims == [
        "Answer cites documents that are not present in evidence."
    ]
    assert result.missing_citations == ["missing-doc"]


def test_grounding_verifier_passes_for_existing_citation() -> None:
    verifier = SimpleGroundingVerifier()

    result = verifier.verify(
        answer="The runtime turns tool results into observations [observation].",
        evidence=_build_evidence(),
    )

    assert result.ok is True
    assert result.unsupported_claims == []
    assert result.missing_citations == []


def test_verify_answer_tool_schema_and_output() -> None:
    tool = create_verifier_tool(SimpleGroundingVerifier())

    assert tool.schema_info() == {
        "name": "verify_answer",
        "description": "Verify that an answer cites available evidence doc_id values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence": {"type": "array"},
            },
            "required": ["answer", "evidence"],
        },
    }

    result = tool.call(
        {
            "answer": "The answer cites the evidence [observation].",
            "evidence": [item.to_dict() for item in _build_evidence()],
        }
    )

    assert result.ok is True
    assert result.output == {
        "ok": True,
        "unsupported_claims": [],
        "missing_citations": [],
        "notes": "Answer includes at least one citation and all citations exist.",
    }


def _build_corpus() -> InMemoryCorpus:
    corpus = InMemoryCorpus()
    corpus.add(
        Document(
            id="agent-loop",
            title="Agent Loop",
            text=(
                "An agent loop lets an LLM decide whether to call a tool or "
                "return a final answer. The runtime executes tool calls."
            ),
        )
    )
    corpus.add(
        Document(
            id="runtime",
            title="Runtime Harness",
            text=(
                "The runtime owns execution. It validates tool calls, records "
                "observations, and enforces max steps."
            ),
        )
    )
    corpus.add(
        Document(
            id="trace",
            title="Trace Recorder",
            text=(
                "Trace records user messages, tool calls, tool observations, "
                "and final answers for debugging."
            ),
        )
    )
    return corpus


def _build_agentic_rag_runtime(
    decisions: list[AssistantDecision],
    max_steps: int,
) -> tuple[AgentRuntime, TraceRecorder]:
    registry = ToolRegistry()
    registry.register(create_retriever_tool(SimpleKeywordRetriever(_build_corpus())))
    trace = TraceRecorder()
    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tool_registry=registry,
        max_steps=max_steps,
        trace_recorder=trace,
    )
    return runtime, trace


def _build_evidence() -> list[Evidence]:
    return [
        Evidence(
            doc_id="agent-loop",
            title="Agent Loop",
            snippet="An agent loop lets an LLM decide whether to call a tool.",
            score=1,
        ),
        Evidence(
            doc_id="observation",
            title="Tool Observation",
            snippet="A tool observation carries tool results back into LLM context.",
            score=2,
        ),
    ]
