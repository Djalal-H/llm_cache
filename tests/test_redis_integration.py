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
