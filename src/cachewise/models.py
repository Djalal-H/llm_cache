from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: NonemptyText = Field(max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: NonemptyText = Field(max_length=80)
    question: NonemptyText = Field(max_length=4000)
    language: NonemptyText = Field(default="en", max_length=40)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)


class Usage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    prefix_cached_tokens: int | None = Field(default=None, ge=0)


class GenerationResult(BaseModel):
    answer: str
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    finish_reason: str | None = None


class EmbeddingResult(BaseModel):
    embedding: list[float]
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None


class JudgeResult(BaseModel):
    passed: bool | None
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None


class ChatResponse(BaseModel):
    answer: str
    request_id: str
    cache_outcome: Literal["bypass", "miss", "exact_hit"] = "bypass"
    usage: Usage
    model: str | None
    finish_reason: str | None
    system_prompt_hash: str
    serialized_prompt_hash: str
    policy_version: int
    catalogue_version: int


class InvalidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["policy", "catalogue", "all"]
