"""Application-cache orchestration; generation remains provider neutral."""

import time

from cachewise.assistant import Assistant, PreparedPrompt, canonical, digest
from cachewise.cache import CacheEntry, CacheError, ExactCache, Namespace
from cachewise.config import Settings
from cachewise.metrics import Metrics
from cachewise.models import ChatRequest, ChatResponse, Usage


def exact_identity(prompt: PreparedPrompt, config: Settings, namespace: Namespace) -> str:
    return digest(
        canonical(
            {
                "schema": 1,
                "namespace": namespace.model_dump(),
                "prompt": prompt.serialized_prompt_hash,
                "system_prompt": prompt.system_prompt_hash,
                "tier": prompt.tier,
                "region": prompt.region,
                "policy": prompt.policy_version,
                "catalogue": prompt.catalogue_version,
                "endpoint": config.generation_base_url,
                "model": config.generation_model,
                "revision": config.generation_model_revision,
                "generation": {"temperature": config.temperature, "max_tokens": config.max_tokens},
                "tools": [],
            }
        )
    )


class ChatService:
    def __init__(
        self, assistant: Assistant, config: Settings, cache: ExactCache | None, metrics: Metrics
    ):
        self.assistant, self.config, self.cache, self.metrics = assistant, config, cache, metrics

    async def answer(
        self, request: ChatRequest, prompt: PreparedPrompt, request_id: str
    ) -> ChatResponse:
        metrics = self.metrics
        identity = None
        outcome = "bypass"
        if self.config.cache_mode == "exact" and prompt.eligibility == "eligible":
            outcome = "miss"
            if self.cache is not None:
                try:
                    namespace = await self.cache.namespace()
                    identity = exact_identity(prompt, self.config, namespace)
                    entry = await self.cache.get(identity)
                    if entry is not None and entry.expires_at > time.time():
                        metrics.exact_hits += 1
                        return ChatResponse(
                            answer=entry.answer,
                            request_id=request_id,
                            cache_outcome="exact_hit",
                            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                            model=entry.model,
                            finish_reason=entry.finish_reason,
                            system_prompt_hash=prompt.system_prompt_hash,
                            serialized_prompt_hash=prompt.serialized_prompt_hash,
                            policy_version=prompt.policy_version,
                            catalogue_version=prompt.catalogue_version,
                        )
                except CacheError:
                    metrics.cache_errors += 1
            metrics.cache_misses += 1
        else:
            reason = "disabled" if self.config.cache_mode == "disabled" else prompt.eligibility
            metrics.bypasses[reason] += 1
        metrics.generation_calls += 1
        response = await self.assistant.answer(request, prompt, request_id)
        metrics.record_usage(response.usage)
        response.cache_outcome = outcome
        if (
            identity is not None
            and self.cache is not None
            and response.answer.strip()
            and response.finish_reason == "stop"
        ):
            entry = CacheEntry(
                identity=identity,
                answer=response.answer,
                model=response.model,
                finish_reason=response.finish_reason,
                expires_at=time.time() + self.config.cache_ttl_seconds,
            )
            try:
                await self.cache.put(entry, self.config.cache_ttl_seconds)
            except CacheError:
                metrics.cache_errors += 1
        return response
