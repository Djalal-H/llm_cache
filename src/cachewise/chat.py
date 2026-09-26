"""Application-cache orchestration; generation remains provider neutral."""

import time

from cachewise.assistant import Assistant, PreparedPrompt, canonical, digest
from cachewise.cache import CacheEntry, CacheError, ExactCache, Namespace
from cachewise.config import LAYA_QUESTION_VERSION, Settings
from cachewise.metrics import Metrics
from cachewise.models import ChatRequest, ChatResponse, Usage
from cachewise.providers import EmbeddingProvider, ProviderError
from cachewise.semantic import SemanticCache, semantic_context, vector_bytes


def exact_identity(prompt: PreparedPrompt, config: Settings, namespace: Namespace) -> str:
    return digest(
        canonical(
            {
                "schema": 1,
                "eligibility": {
                    "mode": config.eligibility_mode,
                    "question_version": LAYA_QUESTION_VERSION
                    if config.eligibility_mode == "laya"
                    else None,
                    "model": config.laya_model if config.eligibility_mode == "laya" else None,
                    "revision": config.laya_model_revision
                    if config.eligibility_mode == "laya"
                    else None,
                    "threshold": config.laya_min_answer_confidence
                    if config.eligibility_mode == "laya"
                    else None,
                },
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
        self,
        assistant: Assistant,
        config: Settings,
        cache: ExactCache | None,
        metrics: Metrics,
        embedding: EmbeddingProvider | None = None,
        semantic: SemanticCache | None = None,
    ):
        self.assistant, self.config, self.cache, self.metrics = assistant, config, cache, metrics
        self.embedding, self.semantic = embedding, semantic

    @staticmethod
    def hit(entry, prompt, request_id, outcome, embedding_usage=None) -> ChatResponse:
        return ChatResponse(
            answer=entry.answer,
            request_id=request_id,
            cache_outcome=outcome,
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            embedding_usage=embedding_usage,
            semantic_similarity=getattr(entry, "similarity", None),
            semantic_source_identity=entry.identity if outcome == "semantic_hit" else None,
            model=entry.model,
            finish_reason=entry.finish_reason,
            system_prompt_hash=prompt.system_prompt_hash,
            serialized_prompt_hash=prompt.serialized_prompt_hash,
            policy_version=prompt.policy_version,
            catalogue_version=prompt.catalogue_version,
        )

    async def answer(
        self, request: ChatRequest, prompt: PreparedPrompt, request_id: str
    ) -> ChatResponse:
        metrics = self.metrics
        identity = namespace = context = vector = embedding_usage = None
        outcome = "bypass"
        if self.config.cache_mode != "disabled" and prompt.eligibility == "eligible":
            outcome = "miss"
            if self.cache is not None:
                try:
                    namespace = await self.cache.namespace()
                    identity = exact_identity(prompt, self.config, namespace)
                    entry = await self.cache.get(identity)
                    if entry is not None and entry.expires_at > time.time():
                        metrics.exact_hits += 1
                        return self.hit(entry, prompt, request_id, "exact_hit")
                except CacheError:
                    metrics.cache_errors += 1
            if (
                self.config.cache_mode == "semantic"
                and namespace is not None
                and self.embedding is not None
                and self.semantic is not None
            ):
                context = semantic_context(prompt, self.config, namespace)
                metrics.embedding_calls += 1
                try:
                    result = await self.embedding.embed(prompt.messages[-1].content)
                    embedding_usage = result.usage
                    metrics.record_embedding(result.usage)
                    vector = vector_bytes(result.embedding, self.config.embedding_dimension)
                except (ProviderError, ValueError, OverflowError):
                    metrics.embedding_errors += 1
                if vector is not None:
                    try:
                        entry = await self.semantic.search(
                            context, vector, self.config.semantic_threshold
                        )
                        if (
                            entry is not None
                            and entry.context == context
                            and entry.expires_at > time.time()
                            and entry.similarity >= self.config.semantic_threshold
                        ):
                            metrics.semantic_hits += 1
                            return self.hit(
                                entry, prompt, request_id, "semantic_hit", embedding_usage
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
        response.embedding_usage = embedding_usage
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
            if vector is not None and self.semantic is not None:
                try:
                    await self.semantic.put(context, vector, entry, self.config.cache_ttl_seconds)
                except CacheError:
                    metrics.cache_errors += 1
        return response
