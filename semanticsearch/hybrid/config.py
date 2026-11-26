
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HybridConfig:
    # Number of candidates collected from each searcher.
    lexical_limit: int = 100
    semantic_limit: int = 100

    # Final number of fused results.
    default_limit: int = 20

    # Reciprocal Rank Fusion parameter.
    rrf_k: int = 60

    # Relative importance of each retrieval method.
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.lexical_limit < 1:
            raise ValueError("lexical_limit must be positive")

        if self.semantic_limit < 1:
            raise ValueError("semantic_limit must be positive")

        if self.default_limit < 1:
            raise ValueError("default_limit must be positive")

        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        if self.lexical_weight < 0:
            raise ValueError("lexical_weight cannot be negative")

        if self.semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative")

        if self.lexical_weight + self.semantic_weight == 0:
            raise ValueError("At least one retrieval weight must be positive")
