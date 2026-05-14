from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_lab.core.tools import Tool
from agent_lab.rag.corpus import Document, InMemoryCorpus


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    title: str
    snippet: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
        }


class SimpleKeywordRetriever:
    def __init__(self, corpus: InMemoryCorpus) -> None:
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 3) -> list[Evidence]:
        query_tokens = _tokenize(query)
        if not query_tokens or top_k <= 0:
            return []

        scored: list[Evidence] = []
        for document in self.corpus.list():
            document_tokens = _tokenize(f"{document.title} {document.text}")
            overlap = query_tokens & document_tokens
            if not overlap:
                continue

            scored.append(
                Evidence(
                    doc_id=document.id,
                    title=document.title,
                    snippet=_make_snippet(document, overlap),
                    score=len(overlap),
                )
            )

        return sorted(scored, key=lambda item: (-item.score, item.doc_id))[:top_k]


def create_retriever_tool(retriever: SimpleKeywordRetriever) -> Tool:
    return Tool(
        name="retrieve_docs",
        description="Retrieve relevant documents from the toy corpus.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        fn=lambda args: [
            evidence.to_dict()
            for evidence in retriever.retrieve(
                query=args["query"],
                top_k=args.get("top_k", 3),
            )
        ],
    )


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _make_snippet(document: Document, overlap: set[str]) -> str:
    text = " ".join(document.text.split())
    lower_text = text.lower()
    first_match = min(
        (lower_text.find(token) for token in overlap if token in lower_text),
        default=0,
    )
    start = max(first_match - 40, 0)
    end = min(first_match + 160, len(text))
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet
