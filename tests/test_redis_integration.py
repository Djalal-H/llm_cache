"""Opt in with CACHEWISE_TEST_REDIS_URL; isolated keys, no generation provider calls."""

import asyncio
import os
import time
from uuid import uuid4

import pytest

from cachewise.cache import CacheEntry, CacheError, RedisExactCache


@pytest.fixture
async def redis_cache():
    url = os.environ.get("CACHEWISE_TEST_REDIS_URL")
    if not url:
        pytest.skip("set CACHEWISE_TEST_REDIS_URL to a dedicated Redis test instance")
    cache = RedisExactCache(url, prefix=f"cachewise-test:{uuid4()}")
    try:
        assert await cache.ready()
        yield cache
    finally:
        keys = [key async for key in cache.client.scan_iter(match=f"{cache.prefix}:*")]
        if keys:
            await cache.client.delete(*keys)
        await cache.close()


def entry(identity="test", ttl=30):
    return CacheEntry(
        identity=identity,
        answer="A generated FAQ",
        model="test",
        finish_reason="stop",
        expires_at=time.time() + ttl,
    )


async def test_real_redis_ttl_hit_and_first_writer(redis_cache):
    cached = entry(ttl=1)
    await redis_cache.put(cached, 1)
    key = f"{redis_cache.prefix}:exact:test"
    before = await redis_cache.client.pttl(key)
    assert (await redis_cache.get("test")).answer == cached.answer
    await redis_cache.put(entry(ttl=60), 60)
    assert await redis_cache.client.pttl(key) <= before
    await asyncio.sleep(1.05)
    assert await redis_cache.get("test") is None
    assert not await redis_cache.client.exists(key)


async def test_real_redis_explicit_expiry_and_corruption(redis_cache):
    await redis_cache.put(entry(ttl=-1), 30)
    assert await redis_cache.get("test") is None
    key = f"{redis_cache.prefix}:exact:bad"
    for raw in ("invalid JSON", entry("other").model_dump_json()):
        await redis_cache.client.set(key, raw)
        with pytest.raises(CacheError):
            await redis_cache.get("bad")


async def test_atomic_versions_and_lost_state(redis_cache):
    snapshots = await asyncio.gather(*(redis_cache.namespace() for _ in range(20)))
    assert all(value == snapshots[0] for value in snapshots)
    await asyncio.gather(*(redis_cache.invalidate("policy") for _ in range(20)))
    state = await redis_cache.namespace()
    assert state.policy == 21 and state.catalogue == 1
    updated = await redis_cache.invalidate("all")
    assert updated.policy == 22 and updated.catalogue == 2
    for raw in (None, "broken", '{"identifier":"old","policy":1}'):
        if raw is None:
            await redis_cache.client.delete(f"{redis_cache.prefix}:namespace")
        else:
            await redis_cache.client.set(f"{redis_cache.prefix}:namespace", raw)
        recovered = await redis_cache.namespace()
        assert recovered.identifier != state.identifier
        assert recovered.policy == recovered.catalogue == 1
        state = recovered


async def test_unavailable_redis_has_bounded_failure(redis_cache):
    # Closing a local connection with an unused port exercises real transport fallback.
    unavailable = RedisExactCache("redis://127.0.0.1:1/0", timeout=0.1)
    try:
        started = time.monotonic()
        assert not await unavailable.ready()
        with pytest.raises(CacheError):
            await unavailable.namespace()
        assert time.monotonic() - started < 1
    finally:
        await unavailable.close()


async def test_real_redis_api_miss_hit_invalidate_miss(redis_cache, generation):
    import httpx

    from cachewise.api import create_app
    from cachewise.config import Settings

    config = Settings(_env_file=None, cache_mode="exact", admin_token="integration-only")
    app = create_app(config, generation, redis_cache)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            body = {"customer_id": "demo-alice", "question": "How much does shipping cost?"}
            first = await client.post("/chat", json=body)
            second = await client.post("/chat", json=body)
            assert first.json()["cache_outcome"] == "miss"
            assert second.json()["cache_outcome"] == "exact_hit"
            invalidation = await client.post(
                "/admin/invalidate",
                json={"scope": "all"},
                headers={"Authorization": "Bearer integration-only"},
            )
            assert invalidation.status_code == 200
            third = await client.post("/chat", json=body)
            assert third.json()["cache_outcome"] == "miss"
            assert len(generation.messages) == 2
            metrics = (await client.get("/metrics")).json()
            assert metrics["generation_calls"] == 2
            assert metrics["avoided_generation_calls"] == 1


async def test_real_vectors_filter_threshold_expiry_and_first_writer(redis_cache):
    from test_semantic import config

    from cachewise.semantic import RedisSemanticCache, vector_bytes

    semantic = RedisSemanticCache(redis_cache, config())
    ctx, other = "a" * 64, "b" * 64
    vector = vector_bytes([1, 0], 2)
    try:
        assert await semantic.ready()
        await semantic.put(other, vector, entry("wrong-context"), 30)
        assert await semantic.search(ctx, vector, 0.95) is None
        cached = entry("vector", ttl=2)
        await semantic.put(ctx, vector, cached, 30)
        key = semantic.prefix + "entry:vector"
        before = await redis_cache.client.pttl(key)
        hit = await semantic.search(ctx, vector, 0.95)
        assert hit.answer == cached.answer and hit.similarity == 1
        assert await semantic.search(ctx, vector_bytes([0, 1], 2), 0.95) is None
        await semantic.put(ctx, vector, entry("vector", ttl=60), 60)
        assert await redis_cache.client.pttl(key) <= before
        # Simulate index metadata outliving the payload's logical expiry.
        expired = entry("vector", ttl=-1)
        await redis_cache.client.hset(key, "entry", expired.model_dump_json())
        assert await semantic.search(ctx, vector, 0.95) is None
        await redis_cache.client.hset(key, "entry", "invalid json")
        with pytest.raises(CacheError):
            await semantic.search(ctx, vector, 0.95)
        await asyncio.sleep(2.05)
        assert not await redis_cache.client.exists(key)
        assert await semantic.search(ctx, vector, 0.95) is None
    finally:
        await redis_cache.client.execute_command("FT.DROPINDEX", semantic.index)


async def test_real_semantic_api_sequence_and_namespace(redis_cache, generation):
    import httpx
    from test_semantic import Embeddings, config

    from cachewise.api import create_app
    from cachewise.semantic import RedisSemanticCache

    settings = config(admin_token="test-only")
    vectors = RedisSemanticCache(redis_cache, settings)
    embeddings = Embeddings()
    app = create_app(settings, generation, redis_cache, embeddings, vectors)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:

                async def ask(text, **kwargs):
                    response = await client.post(
                        "/chat", json={"customer_id": "demo-alice", "question": text, **kwargs}
                    )
                    assert response.status_code == 200, response.text
                    return response.json()

                first = await ask("How much does shipping cost?")
                assert first["cache_outcome"] == "miss"
                assert (await ask("How much does shipping cost?"))["cache_outcome"] == "exact_hit"
                semantic = await ask("What does shipping cost?")
                assert semantic["cache_outcome"] == "semantic_hit"
                assert semantic["embedding_usage"]["total_tokens"] == 5
                assert (await ask("What does shipping cost?", language="fr"))[
                    "cache_outcome"
                ] == "miss"
                for scope in ("policy", "catalogue"):
                    response = await client.post(
                        "/admin/invalidate",
                        json={"scope": scope},
                        headers={"Authorization": "Bearer test-only"},
                    )
                    assert response.status_code == 200
                    assert (await ask("What does shipping cost?"))["cache_outcome"] == "miss"
                health = (await client.get("/health")).json()
                assert health["semantic_cache"] == "ready"
                metrics = (await client.get("/metrics")).json()
                assert metrics["semantic_hits"] == 1
                assert metrics["exact_hits"] == 1
                assert metrics["cache_errors"] == 0
                assert metrics["cache_namespace"] is not None
    finally:
        await redis_cache.client.execute_command("FT.DROPINDEX", vectors.index)
