from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

AskAttribute = Literal[
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
]


class AgentProtocol(Protocol):
    def reset(self, session_id: str, user_profile: dict) -> None: ...

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict: ...


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parent_asin: str = Field(min_length=1)
    score: float | None = None

    @field_validator("parent_asin")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("parent_asin cannot be blank")
        return normalized


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class TurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    ask_attribute: AskAttribute | None
    recommendations: list[Recommendation] = Field(max_length=100)
    usage: Usage = Field(default_factory=Usage)
