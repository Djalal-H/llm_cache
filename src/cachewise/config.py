from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LAYA_QUESTION_VERSION = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CACHEWISE_", env_file=".env", extra="ignore")

    generation_base_url: str = "http://127.0.0.1:8001/v1"
    generation_model: str = "cachewise-model"
    generation_model_revision: str | None = None
    cache_mode: Literal["disabled", "exact", "semantic"] = "disabled"
    eligibility_mode: Literal["lexical", "laya"] = "lexical"
    laya_base_url: str | None = None
    laya_model: Literal["english", "multilingual", "typed-decisions"] = "english"
    laya_model_revision: str | None = None
    laya_api_key: SecretStr = SecretStr("")
    laya_timeout_seconds: float = 5
    laya_min_answer_confidence: float | None = None
    redis_url: SecretStr = SecretStr("redis://127.0.0.1:6379/0")
    cache_ttl_seconds: int = 86400
    cache_timeout_seconds: float = 1
    admin_token: SecretStr = SecretStr("")
    generation_api_key: SecretStr = SecretStr("")
    generation_timeout_seconds: float = 120
    temperature: float = 0
    max_tokens: int = 256
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_model_revision: str | None = None
    semantic_threshold: float | None = None
    embedding_api_key: SecretStr = SecretStr("")
    embedding_timeout_seconds: float = 30
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: SecretStr = SecretStr("")
    judge_timeout_seconds: float = 120
    fixture_path: Path | None = None
    vllm_metrics_url: str = "http://127.0.0.1:8001/metrics"

    @field_validator(
        "embedding_base_url",
        "embedding_model",
        "judge_base_url",
        "judge_model",
        "generation_model_revision",
        "embedding_model_revision",
        "laya_base_url",
        "laya_model_revision",
    )
    @classmethod
    def blank_optional(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator(
        "generation_base_url",
        "embedding_base_url",
        "judge_base_url",
        "laya_base_url",
        "vllm_metrics_url",
    )
    @classmethod
    def http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider URLs must be absolute HTTP(S) URLs")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "credentials, query strings and fragments do not belong in provider URLs"
            )
        return value.rstrip("/")

    @field_validator("generation_model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("generation model must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if self.embedding_dimension is not None and not 1 <= self.embedding_dimension <= 65536:
            raise ValueError("embedding dimension must be in [1, 65536]")
        if self.semantic_threshold is not None and not 0.80 <= self.semantic_threshold <= 0.99:
            raise ValueError("semantic threshold must be in [0.80, 0.99]")
        if self.cache_mode == "semantic" and (
            not self.embedding_base_url
            or not self.embedding_model
            or self.embedding_dimension is None
            or self.semantic_threshold is None
        ):
            raise ValueError(
                "semantic mode requires embedding endpoint, model, dimension and threshold"
            )
        if self.eligibility_mode == "laya":
            if self.laya_base_url is None or self.laya_min_answer_confidence is None:
                raise ValueError("Laya mode requires base URL and an explicit confidence threshold")
            if self.cache_mode == "semantic":
                raise ValueError("Laya eligibility is limited to disabled or exact cache mode")
        if self.laya_min_answer_confidence is not None and not (
            0 < self.laya_min_answer_confidence <= 1
        ):
            raise ValueError("Laya confidence threshold must be in (0, 1]")
        if not 0 < self.laya_timeout_seconds <= 60:
            raise ValueError("Laya timeout must be in (0, 60]")
        if not 1 <= self.cache_ttl_seconds <= 604800:
            raise ValueError("cache TTL must be between one second and seven days")
        if not 0 < self.cache_timeout_seconds <= 10:
            raise ValueError("cache timeout must be in (0, 10]")
        parsed = urlsplit(self.redis_url.get_secret_value())
        if (
            parsed.scheme not in {"redis", "rediss"}
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
            or (parsed.path and not parsed.path.lstrip("/").isdigit())
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError("Redis URL must use redis(s), without query or fragment")
        for name in ("generation", "embedding", "judge"):
            timeout = getattr(self, f"{name}_timeout_seconds")
            if not 0 < timeout <= 600:
                raise ValueError(f"{name} timeout must be in (0, 600]")
        for name in ("embedding", "judge"):
            if bool(getattr(self, f"{name}_base_url")) != bool(getattr(self, f"{name}_model")):
                raise ValueError(f"{name} base URL and model must be configured together")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")
        if not 1 <= self.max_tokens <= 2048:
            raise ValueError("max_tokens must be in [1, 2048]")
        return self

    def public_config(self) -> dict:
        # No endpoint credentials, SecretStr fields, or fixture paths in exported results.
        return {
            "generation_model": self.generation_model,
            "generation_timeout_seconds": self.generation_timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "application_cache": self.cache_mode,
            "eligibility_mode": self.eligibility_mode,
            "laya_model": self.laya_model if self.eligibility_mode == "laya" else None,
            "laya_model_revision": self.laya_model_revision
            if self.eligibility_mode == "laya"
            else None,
            "laya_question_version": LAYA_QUESTION_VERSION
            if self.eligibility_mode == "laya"
            else None,
            "laya_min_answer_confidence": self.laya_min_answer_confidence
            if self.eligibility_mode == "laya"
            else None,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_timeout_seconds": self.cache_timeout_seconds,
            "generation_model_revision": self.generation_model_revision,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_dimension": self.embedding_dimension,
            "semantic_threshold": self.semantic_threshold,
            "judge_model": self.judge_model,
            "generation_dollar_estimate": None,
            "generation_cost_assumption": "No self-hosted serving-cost model configured",
        }
