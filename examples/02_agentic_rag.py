from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_lab.core.runtime import AgentRuntime
from agent_lab.core.tools import ToolRegistry
from agent_lab.core.trace import TraceRecorder
from agent_lab.llm.base import AssistantDecision
from agent_lab.llm.fake_llm import FakeLLM
from agent_lab.rag.corpus import Document, InMemoryCorpus
from agent_lab.rag.retriever import SimpleKeywordRetriever, create_retriever_tool
from agent_lab.rag.verifier import SimpleGroundingVerifier, create_verifier_tool


VERIFY_EVIDENCE = [
    {
        "doc_id": "agent-loop",
        "title": "Agent Loop",
        "snippet": (
            "An agent loop lets an LLM decide whether to call a tool or "
            "return a final answer. The runtime executes tool calls."
        ),
        "score": 1,
    },
    {
        "doc_id": "observation",
        "title": "Tool Observation",
        "snippet": (
            "A tool observation is the message that carries a tool result "
            "back into the LLM context."
        ),
        "score": 4,
    },
]


def main() -> None:
    corpus = build_toy_corpus()
    retriever = SimpleKeywordRetriever(corpus)

    registry = ToolRegistry()
    registry.register(create_retriever_tool(retriever))
    registry.register(create_verifier_tool(SimpleGroundingVerifier()))

    llm = FakeLLM(
        [
            AssistantDecision.tool_call(
                "retrieve_docs",
                {"query": "agent", "top_k": 2},
            ),
            AssistantDecision.tool_call(
                "retrieve_docs",
                {"query": "tool observation returned to LLM context", "top_k": 2},
            ),
            AssistantDecision.tool_call(
                "verify_answer",
                {
                    "answer": (
                        "Agent runtime executes tool calls and turns results "
                        "into observations before the LLM continues."
                    ),
                    "evidence": VERIFY_EVIDENCE,
                },
            ),
            AssistantDecision.tool_call(
                "verify_answer",
                {
                    "answer": (
                        "Agent runtime executes tool calls and turns results "
                        "into observations before the LLM continues. Evidence: "
                        "[agent-loop], [observation]."
                    ),
                    "evidence": VERIFY_EVIDENCE,
                },
            ),
            AssistantDecision.final(
                "Agent runtime executes tool calls and turns results into "
                "observations before the LLM continues. Evidence: "
                "[agent-loop], [observation]."
            ),
        ]
    )
    trace = TraceRecorder()
    runtime = AgentRuntime(
        llm=llm,
        tool_registry=registry,
        max_steps=5,
        trace_recorder=trace,
    )

    result = runtime.run(
        "How does an agent runtime use tool observations? Cite doc_id values."
    )

    print(trace.as_text())
    print(f"final_ok={result.ok} output={result.output} error={result.error}")


def build_toy_corpus() -> InMemoryCorpus:
    corpus = InMemoryCorpus()
    corpus.add(
        Document(
            id="agent-loop",
            title="Agent Loop",
            text=(
                "An agent loop lets an LLM decide whether to call a tool or "
                "return a final answer. The runtime executes tool calls."
            ),
            metadata={"topic": "agents"},
        )
    )
    corpus.add(
        Document(
            id="tool-registry",
            title="Tool Registry",
            text=(
                "A tool registry stores tool names, descriptions, input schemas, "
                "and callables so the runtime can execute a selected tool."
            ),
            metadata={"topic": "tools"},
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
            metadata={"topic": "observability"},
        )
    )
    corpus.add(
        Document(
            id="observation",
            title="Tool Observation",
            text=(
                "A tool observation is the message that carries a tool result "
                "back into the LLM context."
            ),
            metadata={"topic": "messages"},
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
            metadata={"topic": "runtime"},
        )
    )
    corpus.add(
        Document(
            id="fake-llm",
            title="Fake LLM",
            text=(
                "A fake LLM returns scripted decisions, making agent loop tests "
                "deterministic and reproducible."
            ),
            metadata={"topic": "testing"},
        )
    )
    return corpus


if __name__ == "__main__":
    main()
