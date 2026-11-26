from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticSearchResult:
    query: str
    hits: list[Any] = field(default_factory=list)

    elapsed_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.hits)

    def all_hits(self) -> list[Any]:
        return self.hits

    def episodes(self) -> list[Any]:
        """Group results by episode using the lexical search helpers."""
        from semanticsearch.lexical.search import group_by_episode

        return group_by_episode(self.hits)