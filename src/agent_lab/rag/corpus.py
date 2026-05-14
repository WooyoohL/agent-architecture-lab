from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryCorpus:
    def __init__(self) -> None:
        self._documents: list[Document] = []

    def add(self, document: Document) -> None:
        self._documents.append(document)

    def list(self) -> list[Document]:
        return list(self._documents)
