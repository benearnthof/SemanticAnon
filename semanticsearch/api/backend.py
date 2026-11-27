from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .schemas import EpisodeResponse


@dataclass(frozen=True)
class SearchOptions:
    limit: int
    lexical_weight: float
    semantic_weight: float


class SearchBackend(Protocol):
    """Small adapter boundary between FastAPI and the existing search engine.

    Implement this against HybridSearcher so HTTP concerns do not leak into
    the retrieval implementation. The backend instance is created once at
    application startup and reused for all requests.
    """

    def search(self, query: str, options: SearchOptions) -> Sequence[EpisodeResponse]: ...

    def close(self) -> None: ...
