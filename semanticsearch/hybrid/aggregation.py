
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EpisodeHit:
    episode_id: str
    episode_title: str

    score: float
    hits: list[Any] = field(default_factory=list)

    @property
    def best_hit(self) -> Any:
        return self.hits[0]

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    @property
    def seek_points(self) -> list[float]:
        return sorted({h.hit.seek_s for h in self.hits})


def aggregate_episodes(
    hits: list[Any],
    *,
    support_weight: float = 0.25,
    support_limit: int = 5,
    limit: int | None = None,
) -> list[EpisodeHit]:
    """
    Aggregate passage-level HybridHits into ranked episodes.

    The highest-scoring passage determines the base episode score.
    Additional high-ranking passages contribute diminishing returns.

    Passage order inside each episode is retained by score.
    """
    if support_weight < 0:
        raise ValueError("support_weight cannot be negative")

    if support_limit < 1:
        raise ValueError("support_limit must be positive")

    groups: dict[str, list[Any]] = {}

    for hit in hits:
        episode_id = hit.hit.episode_id
        groups.setdefault(episode_id, []).append(hit)

    episodes = []

    for episode_id, episode_hits in groups.items():
        ranked = sorted(
            episode_hits,
            key=lambda h: h.score,
            reverse=True,
        )

        selected = ranked[:support_limit]

        # Best passage contributes its full score.
        score = selected[0].score

        # Additional passages provide diminishing support.
        for rank, hit in enumerate(selected[1:], start=2):
            score += support_weight * hit.score / rank

        episodes.append(
            EpisodeHit(
                episode_id=episode_id,
                episode_title=selected[0].hit.episode_title,
                score=score,
                hits=selected,
            )
        )

    episodes.sort(
        key=lambda e: e.score,
        reverse=True,
    )

    if limit is not None:
        episodes = episodes[:limit]

    return episodes
