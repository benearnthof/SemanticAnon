"""Adapter connecting the FastAPI layer to the existing search package."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Sequence

from semanticsearch.lexical.search import LexicalSearcher
from semanticsearch.hybrid.config import HybridConfig
from semanticsearch.hybrid.searcher import HybridSearcher

from .backend import SearchBackend, SearchOptions
from .schemas import EpisodeResponse, PassageResponse


class PodcastSearchBackend(SearchBackend):
    """Own one lexical connection and one persistent hybrid/ONNX searcher."""

    def __init__(self) -> None:
        self.lexical = LexicalSearcher()
        try:
            self.hybrid = HybridSearcher(
                self.lexical,
                config=HybridConfig(),
            )
        except Exception:
            self.lexical.close()
            raise

        # FastAPI also serializes calls. This lock makes the adapter safe if it
        # is reused outside that particular application entry point.
        self._lock = threading.Lock()
        self._closed = False

    def search(self, query: str, options: SearchOptions) -> Sequence[EpisodeResponse]:
        if self._closed:
            raise RuntimeError("Search backend is closed")

        with self._lock:
            # Retrieve a full candidate pool, then group it by episode. This
            # avoids relying on the optional EpisodeHit aggregation structure
            # and preserves the RRF ranking of each passage.
            candidate_limit = (
                self.hybrid.config.lexical_limit
                + self.hybrid.config.semantic_limit
            )
            result = self.hybrid.search(
                query,
                limit=candidate_limit,
                episodes=False,
                lexical_weight=options.lexical_weight,
                semantic_weight=options.semantic_weight,
            )

        grouped: OrderedDict[str, dict] = OrderedDict()
        for ranked_hit in result.hits:
            hit = ranked_hit.hit
            episode = grouped.setdefault(
                hit.episode_id,
                {
                    "episode_title": hit.episode_title,
                    "passages": [],
                },
            )
            episode["passages"].append(
                PassageResponse(
                    text=hit.text,
                    start_s=hit.start_s,
                    end_s=hit.end_s,
                )
            )

        responses: list[EpisodeResponse] = []
        for episode_id, episode in list(grouped.items())[: options.limit]:
            passages = episode["passages"]
            if not passages:
                continue
            responses.append(
                EpisodeResponse(
                    episode_id=episode_id,
                    episode_title=episode["episode_title"],
                    best_passage=passages[0],
                    additional_passages=passages[1:],
                )
            )
        return responses

    def close(self) -> None:
        if self._closed:
            return
        # HybridSearcher owns/closes the SemanticSearcher, but deliberately
        # does not close the LexicalSearcher whose SQLite connection it uses.
        try:
            self.hybrid.close()
        finally:
            self.lexical.close()
            self._closed = True


def build_backend() -> SearchBackend:
    """Factory called once by FastAPI's lifespan handler."""
    return PodcastSearchBackend()
