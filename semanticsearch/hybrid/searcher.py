
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from semanticsearch.lexical.search import LexicalSearcher
from semanticsearch.semantic.config import SemanticConfig
from semanticsearch.semantic.searcher import SemanticSearcher

from semanticsearch.hybrid.config import HybridConfig
from semanticsearch.hybrid.fusion import (
    HybridHit,
    reciprocal_rank_fusion,
)

from semanticsearch.hybrid.aggregation import (
    EpisodeHit,
    aggregate_episodes,
)


@dataclass(slots=True)
class HybridSearchResult:
    query: str
    hits: list[HybridHit] = field(default_factory=list)
    episode_hits: list[EpisodeHit] = field(default_factory=list)

    elapsed_ms: float = 0.0
    lexical_ms: float = 0.0
    semantic_ms: float = 0.0

    lexical_candidates: int = 0
    semantic_candidates: int = 0

    @property
    def total(self) -> int:
        return len(self.episode_hits) if self.episode_hits else len(self.hits)

    def all_hits(self) -> list[Any]:
        return [h.hit for h in self.hits]


class HybridSearcher:
    def __init__(
        self,
        lexical: LexicalSearcher,
        *,
        config: HybridConfig | None = None,
        semantic_config: SemanticConfig | None = None,
    ) -> None:
        self.lexical = lexical
        self.config = config or HybridConfig()

        # SemanticSearcher reuses the lexical SQLite connection.
        self.semantic = SemanticSearcher(
            lexical,
            config=semantic_config,
        )

    def close(self) -> None:
        self.semantic.close()

    def __enter__(self) -> "HybridSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        episodes: bool = False,
        support_weight: float = 0.25,
        support_limit: int = 5,
        lexical_weight: float | None = None,
        semantic_weight: float | None = None,
    ) -> HybridSearchResult:
        """
        Run lexical and semantic retrieval, then fuse their rankings.
        """
        t0 = time.perf_counter()

        final_limit = (
            self.config.default_limit
            if limit is None
            else limit
        )

        if final_limit < 1:
            raise ValueError("limit must be positive")

        # 1. Lexical retrieval.
        lexical_result = self.lexical.search(
            query,
            limit=self.config.lexical_limit,
            lexical_limit=self.config.lexical_limit,
        )

        lexical_hits = lexical_result.all_hits()

        # 2. Semantic retrieval.
        semantic_result = self.semantic.search(
            query,
            limit=self.config.semantic_limit,
        )

        semantic_hits = semantic_result.hits

        # 3. Reciprocal Rank Fusion.
        # Per-query weights override the defaults without mutating shared config.
        # This allows concurrent callers to choose different balances safely.
        effective_lexical_weight = (
            self.config.lexical_weight
            if lexical_weight is None
            else lexical_weight
        )
        effective_semantic_weight = (
            self.config.semantic_weight
            if semantic_weight is None
            else semantic_weight
        )
        if effective_lexical_weight < 0 or effective_semantic_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if effective_lexical_weight + effective_semantic_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")

        fused = reciprocal_rank_fusion(
            lexical_hits,
            semantic_hits,
            k=self.config.rrf_k,
            lexical_weight=effective_lexical_weight,
            semantic_weight=effective_semantic_weight,
        )

        episode_hits = []

        if episodes:
            episode_hits = aggregate_episodes(
                fused,
                support_weight=support_weight,
                support_limit=support_limit,
                limit=final_limit,
            )
            passage_hits = []
        else:
            passage_hits = fused[:final_limit]

        elapsed = (time.perf_counter() - t0) * 1000

        return HybridSearchResult(
            query=query,
            hits=passage_hits,
            episode_hits=episode_hits,
            elapsed_ms=elapsed,
            lexical_ms=lexical_result.elapsed_ms,
            semantic_ms=semantic_result.elapsed_ms,
            lexical_candidates=len(lexical_hits),
            semantic_candidates=len(semantic_hits),
        )