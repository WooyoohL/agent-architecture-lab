from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_lab.core.tools import Tool
from agent_lab.rag.retriever import Evidence


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    unsupported_claims: list[str]
    missing_citations: list[str]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "unsupported_claims": self.unsupported_claims,
            "missing_citations": self.missing_citations,
            "notes": self.notes,
        }


class SimpleGroundingVerifier:
    def verify(self, answer: str, evidence: list[Evidence]) -> VerificationResult:
        cited_doc_ids = _extract_citations(answer)
        available_doc_ids = {item.doc_id for item in evidence}
        missing_citations = [
            doc_id for doc_id in cited_doc_ids if doc_id not in available_doc_ids
        ]
        unsupported_claims: list[str] = []

        if not cited_doc_ids:
            unsupported_claims.append("Answer does not include any evidence citation.")

        if missing_citations:
            unsupported_claims.append(
                "Answer cites documents that are not present in evidence."
            )

        if unsupported_claims:
            return VerificationResult(
                ok=False,
                unsupported_claims=unsupported_claims,
                missing_citations=missing_citations,
                notes="Answer is not grounded in the provided evidence.",
            )

        return VerificationResult(
            ok=True,
            unsupported_claims=[],
            missing_citations=[],
            notes="Answer includes at least one citation and all citations exist.",
        )


def create_verifier_tool(verifier: SimpleGroundingVerifier) -> Tool:
    return Tool(
        name="verify_answer",
        description="Verify that an answer cites available evidence doc_id values.",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence": {"type": "array"},
            },
            "required": ["answer", "evidence"],
        },
        fn=lambda args: verifier.verify(
            answer=args["answer"],
            evidence=[_evidence_from_dict(item) for item in args["evidence"]],
        ).to_dict(),
    )


def _extract_citations(answer: str) -> list[str]:
    return re.findall(r"\[([A-Za-z0-9_-]+)\]", answer)


def _evidence_from_dict(item: dict[str, Any]) -> Evidence:
    return Evidence(
        doc_id=item["doc_id"],
        title=item["title"],
        snippet=item["snippet"],
        score=item["score"],
    )
