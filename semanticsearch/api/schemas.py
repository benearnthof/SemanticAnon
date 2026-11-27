from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2_000)
    # The UI sends one value. The complementary weight is derived server-side.
    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=25)

    @field_validator("query")
    @classmethod
    def query_must_contain_non_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @property
    def semantic_weight(self) -> float:
        return 1.0 - self.lexical_weight


class PassageResponse(BaseModel):
    text: str
    start_s: float | None = None
    end_s: float | None = None


class EpisodeResponse(BaseModel):
    episode_id: str
    episode_title: str
    best_passage: PassageResponse
    additional_passages: list[PassageResponse] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    lexical_weight: float
    semantic_weight: float
    limit: int
    results: list[EpisodeResponse]
    elapsed_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
