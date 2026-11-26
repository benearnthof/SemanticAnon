
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HybridHit:
    hit: Any

    score: float

    lexical_rank: int | None = None
    semantic_rank: int | None = None

    lexical_contribution: float = 0.0
    semantic_contribution: float = 0.0

    @property
    def sources(self) -> tuple[str, ...]:
        sources = []

        if self.lexical_rank is not None:
            sources.append("lexical")

        if self.semantic_rank is not None:
            sources.append("semantic")

        return tuple(sources)


def hit_key(hit: Any) -> tuple:
    """
    Stable passage identity shared by lexical and semantic results.

    passage_id is deliberately avoided because lexical and semantic
    hydration can use different database identifiers.
    """
    ordinals = tuple(getattr(hit, "ordinals", ()) or ())

    if ordinals:
        return (
            hit.episode_id,
            ordinals,
        )

    passage_ids = tuple(getattr(hit, "passage_ids", ()) or ())

    if passage_ids:
        return (
            hit.episode_id,
            passage_ids,
        )

    return (
        hit.episode_id,
        hit.passage_id,
    )


def deduplicate(hits: list[Any]) -> list[Any]:
    """
    Remove duplicate passages while preserving ranking order.
    """
    seen = set()
    result = []

    for hit in hits:
        key = hit_key(hit)

        if key in seen:
            continue

        seen.add(key)
        result.append(hit)

    return result


def reciprocal_rank_fusion(
    lexical_hits: list[Any],
    semantic_hits: list[Any],
    *,
    k: int = 60,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    limit: int | None = None,
) -> list[HybridHit]:
    """
    Fuse two ranked result lists using Reciprocal Rank Fusion.

    Rankings are one-indexed.

    When the same passage appears in both lists, its contributions
    are accumulated. The lexical Hit is retained as the representative
    object when available, preserving lexical match metadata.
    """
    if k < 1:
        raise ValueError("k must be positive")

    if lexical_weight < 0 or semantic_weight < 0:
        raise ValueError("Weights cannot be negative")

    if lexical_weight + semantic_weight == 0:
        raise ValueError("At least one weight must be positive")

    fused: dict[tuple, HybridHit] = {}

    # Process lexical ranking.
    for rank, hit in enumerate(deduplicate(lexical_hits), 1):
        key = hit_key(hit)
        contribution = lexical_weight / (k + rank)

        entry = fused.get(key)

        if entry is None:
            entry = HybridHit(
                hit=hit,
                score=0.0,
            )
            fused[key] = entry

        entry.lexical_rank = rank
        entry.lexical_contribution = contribution
        entry.score += contribution

    # Process semantic ranking.
    for rank, hit in enumerate(deduplicate(semantic_hits), 1):
        key = hit_key(hit)
        contribution = semantic_weight / (k + rank)

        entry = fused.get(key)

        if entry is None:
            entry = HybridHit(
                hit=hit,
                score=0.0,
            )
            fused[key] = entry

        # Preserve the lexical Hit if one already exists.
        entry.semantic_rank = rank
        entry.semantic_contribution = contribution
        entry.score += contribution

    results = sorted(
        fused.values(),
        key=lambda h: h.score,
        reverse=True,
    )

    if limit is not None:
        results = results[:limit]

    return results
