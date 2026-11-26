
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semanticsearch.lexical.search import LexicalSearcher


@dataclass(slots=True)
class ExpandedHit:
    result: Any
    before: list[Any] = field(default_factory=list)
    after: list[Any] = field(default_factory=list)


def expand_context(
    hit: Any,
    lexical: LexicalSearcher,
    *,
    before: int = 1,
    after: int = 1,
) -> ExpandedHit:
    """
    Retrieve neighbouring transcript passages around a hybrid result.
    The original result and its ranking remain unchanged.
    """
    if before < 0 or after < 0:
        raise ValueError("Context sizes cannot be negative")

    if before == 0 and after == 0:
        return ExpandedHit(result=hit)

    neighbours = lexical.context(
        hit.hit,
        before=before,
        after=after,
    )

    first_ordinal = hit.hit.ordinals[0]
    last_ordinal = hit.hit.ordinals[-1]

    preceding = [
        h for h in neighbours
        if h.ordinals[-1] < first_ordinal
    ]

    following = [
        h for h in neighbours
        if h.ordinals[0] > last_ordinal
    ]

    return ExpandedHit(
        result=hit,
        before=preceding,
        after=following,
    )
