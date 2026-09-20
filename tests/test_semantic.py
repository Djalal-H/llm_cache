import struct
import time
from dataclasses import replace

import pytest
from test_cache import MemoryCache, answer, question

from cachewise.assistant import Assistant
from cachewise.cache import CacheError
from cachewise.chat import ChatService
from cachewise.config import Settings
from cachewise.fixtures import FixtureService
from cachewise.metrics import Metrics
from cachewise.models import EmbeddingResult, Usage
from cachewise.providers import ProviderError
from cachewise.semantic import SemanticHit, embedding_namespace, semantic_context, vector_bytes


def config(**kwargs):
    values = dict(
        cache_mode="semantic",
        embedding_base_url="http://embedding.test/v1",
        embedding_model="test",
        embedding_dimension=2,
        semantic_threshold=0.95,
    )
    values.update(kwargs)
    return Settings(_env_file=None, **values)


class Embeddings:
    def __init__(self):
        self.calls = []
        self.vector = [1.0, 0.0]
        self.error = None

    async def embed(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return EmbeddingResult(embedding=self.vector, usage=Usage(total_tokens=5))


class Vectors:
    def __init__(self):
        self.entries = []
        self.error = False
        self.score = 1.0

    async def ready(self):
        return not self.error

    async def search(self, context, vector, threshold):
        if self.error:
            raise CacheError("test_error")
        for ctx, _, entry in self.entries:
            if ctx == context and entry.expires_at > time.time() and self.score >= threshold:
                return SemanticHit(**entry.model_dump(), context=ctx, similarity=self.score)

    async def put(self, context, vector, entry, ttl):
        if self.error:
            raise CacheError("test_error")
        self.entries.append((context, vector, entry))


def service(generation, **kwargs):
    return ChatService(
        Assistant(FixtureService(), generation),
        config(**kwargs),
        MemoryCache(),
        Metrics("semantic"),
        Embeddings(),
        Vectors(),
    )


async def test_exact_first_semantic_hit_no_promotion_or_refresh(generation):
    chat = service(generation)
    first = await answer(chat)
    exact = await answer(chat)
    paraphrase = question("What does shipping cost?")
    semantic = await answer(chat, paraphrase)
    again = await answer(chat, paraphrase)
    assert [r.cache_outcome for r in (first, exact, semantic, again)] == [
        "miss",
        "exact_hit",
        "semantic_hit",
        "semantic_hit",
    ]
    assert len(chat.embedding.calls) == 3
    assert len(generation.messages) == len(chat.semantic.entries) == chat.cache.writes == 1
    assert semantic.semantic_similarity == 1
    assert semantic.semantic_source_identity in chat.cache.entries
    assert semantic.usage.total_tokens == 0 and semantic.usage.prefix_cached_tokens is None
    assert semantic.embedding_usage.total_tokens == 5
    assert semantic.serialized_prompt_hash != first.serialized_prompt_hash
    stats = chat.metrics.snapshot()
    assert stats["avoided_generation_calls"] == 3
    assert stats["embedding_usage"]["known_total_tokens"] == 15
    assert stats["generation_calls"] == stats["cache_misses"] == 1


@pytest.mark.parametrize(
    "first,second",
    [
        ("Can I return a used item?", "Can I return an unused item?"),
        ("Can I return a personalized item?", "Can I return a defective personalized item?"),
        ("Can I return an item?", "Can I not return an item?"),
        ("What are shipping rules to Canada?", "What are shipping rules to Japan?"),
        ("Can I return an item in 30 days?", "Can I return an item in 60 days?"),
    ],
)
async def test_adversarial_filters_even_with_identical_vectors(generation, first, second):
    chat = service(generation)
    assert (await answer(chat, question(first))).cache_outcome == "miss"
    assert (await answer(chat, question(second))).cache_outcome == "miss"
    assert len(generation.messages) == 2


@pytest.mark.parametrize("failure", ["provider", "dimension", "zero", "nan", "redis"])
async def test_failure_preserves_generation_and_exact(generation, failure):
    chat = service(generation)
    if failure == "provider":
        chat.embedding.error = ProviderError("unavailable")
    elif failure == "redis":
        chat.semantic.error = True
    else:
        chat.embedding.vector = {"dimension": [1], "zero": [0, 0], "nan": [float("nan"), 1]}[
            failure
        ]
    assert (await answer(chat)).cache_outcome == "miss"
    assert (await answer(chat)).cache_outcome == "exact_hit"
    assert len(generation.messages) == 1
    assert not chat.semantic.entries
    assert chat.metrics.cache_errors if failure == "redis" else chat.metrics.embedding_errors


async def test_threshold_expiry_invalidation_and_bypass(generation):
    chat = service(generation)
    await answer(chat)
    chat.semantic.score = 0.94
    assert (await answer(chat, question("What does shipping cost?"))).cache_outcome == "miss"
    chat.semantic.score = 1
    for entry in chat.cache.entries.values():
        entry.expires_at = time.time() - 1
    assert (await answer(chat)).cache_outcome == "miss"
    await chat.cache.invalidate("catalogue")
    assert (await answer(chat, question("What does shipping cost?"))).cache_outcome == "miss"
    count = len(chat.embedding.calls)
    assert (await answer(chat, question("Where is my order?"))).cache_outcome == "bypass"
    assert len(chat.embedding.calls) == count


async def test_failed_generation_does_not_store(generation):
    chat = service(generation)
    generation.error = ProviderError("failed")
    with pytest.raises(ProviderError):
        await answer(chat)
    assert not chat.cache.entries and not chat.semantic.entries


def test_context_and_embedding_identity(generation):
    chat = service(generation)
    prompt = chat.assistant.prepare(question())
    original = semantic_context(prompt, chat.config, chat.cache.state)
    paraphrase = chat.assistant.prepare(question("What does shipping cost?"))
    assert semantic_context(paraphrase, chat.config, chat.cache.state) == original
    for change in (
        {"tier": "premium"},
        {"region": "EU"},
        {"policy_version": 99},
        {"catalogue_version": 99},
        {"system_prompt_hash": "new"},
    ):
        assert (
            semantic_context(replace(prompt, **change), chat.config, chat.cache.state) != original
        )
    language = chat.assistant.prepare(question(language="fr"))
    assert semantic_context(language, chat.config, chat.cache.state) != original
    prompt.messages[0] = prompt.messages[0].model_copy(update={"content": "new instructions"})
    assert semantic_context(prompt, chat.config, chat.cache.state) != original
    for change in (
        {"generation_model": "new"},
        {"generation_model_revision": "v2"},
        {"max_tokens": 128},
        {"generation_base_url": "http://other/v1"},
    ):
        assert semantic_context(paraphrase, config(**change), chat.cache.state) != original
    namespace = embedding_namespace(chat.config)
    for change in (
        {"embedding_model": "new"},
        {"embedding_model_revision": "v2"},
        {"embedding_dimension": 3},
        {"embedding_base_url": "http://other/v1"},
    ):
        assert embedding_namespace(config(**change)) != namespace


@pytest.mark.parametrize(
    "change",
    [
        {"embedding_dimension": None},
        {"embedding_dimension": 0},
        {"semantic_threshold": None},
        {"semantic_threshold": float("nan")},
        {"semantic_threshold": 0.79},
        {"semantic_threshold": 1},
        {"embedding_base_url": None, "embedding_model": None},
    ],
)
def test_requires_explicit_valid_configuration(change):
    with pytest.raises(ValueError):
        config(**change)


def test_vector_normalization():
    assert struct.unpack("<2f", vector_bytes([1e300, 0], 2)) == (1, 0)
    assert struct.unpack("<2f", vector_bytes([1e-300, 0], 2)) == (1, 0)


async def test_inflight_invalidation_retains_captured_namespace(generation):
    chat = service(generation)
    generate = generation.generate

    async def invalidate_during_generation(messages):
        await chat.cache.invalidate("all")
        return await generate(messages)

    generation.generate = invalidate_during_generation
    await answer(chat)
    generation.generate = generate
    assert (await answer(chat, question("What does shipping cost?"))).cache_outcome == "miss"
    assert len(generation.messages) == 2


async def test_truncated_answer_never_enters_either_cache(generation):
    chat = service(generation)
    generate = generation.generate

    async def truncated(messages):
        result = await generate(messages)
        result.finish_reason = "length"
        return result

    generation.generate = truncated
    await answer(chat)
    assert not chat.cache.entries and not chat.semantic.entries


async def test_namespace_outage_skips_embeddings(generation):
    chat = service(generation)
    chat.cache.fail.add("namespace")
    assert (await answer(chat)).cache_outcome == "miss"
    assert not chat.embedding.calls and not chat.semantic.entries
    assert len(generation.messages) == 1


async def test_unknown_embedding_usage_stays_unavailable(generation):
    chat = service(generation)

    async def embed(text):
        return EmbeddingResult(embedding=[1, 0])

    chat.embedding.embed = embed
    await answer(chat)
    stats = chat.metrics.snapshot()["embedding_usage"]
    assert stats == {"known_total_tokens": None, "observed_requests": 0, "unavailable_requests": 1}
