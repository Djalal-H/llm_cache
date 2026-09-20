"""Redis exact entries and atomic, recoverable namespace versioning."""

import asyncio
import time
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

Scope = Literal["policy", "catalogue", "all"]


class CacheError(Exception):
    """Cache failure with no endpoint credentials in its public message."""


class Namespace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    identifier: str = Field(min_length=1)
    policy: int = Field(ge=1)
    catalogue: int = Field(ge=1)


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    identity: str
    answer: str = Field(min_length=1)
    model: str | None
    finish_reason: str | None
    expires_at: float = Field(gt=0, allow_inf_nan=False)


class ExactCache(Protocol):
    async def namespace(self) -> Namespace: ...
    async def invalidate(self, scope: Scope) -> Namespace: ...
    async def get(self, identity: str) -> CacheEntry | None: ...
    async def put(self, entry: CacheEntry, ttl: int) -> None: ...
    async def ready(self) -> bool: ...
    async def close(self) -> None: ...


# One persistent JSON value prevents partial counter loss. A missing/corrupt value
# gets a fresh UUID, so surviving entries in the old namespace stay unreachable.
NAMESPACE_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
local state = nil
if raw then
    local ok, value = pcall(cjson.decode, raw)
    if ok and type(value) == 'table' and type(value.identifier) == 'string'
       and string.len(value.identifier) > 0 and type(value.policy) == 'number'
       and type(value.catalogue) == 'number' and value.policy >= 1
       and value.catalogue >= 1 and value.policy == math.floor(value.policy)
       and value.catalogue == math.floor(value.catalogue) then
        state = value
    end
end
if not state then
    state = {identifier=ARGV[1], policy=1, catalogue=1}
end
if ARGV[2] == 'policy' or ARGV[2] == 'all' then state.policy = state.policy + 1 end
if ARGV[2] == 'catalogue' or ARGV[2] == 'all' then
    state.catalogue = state.catalogue + 1
end
local encoded = cjson.encode(state)
redis.call('SET', KEYS[1], encoded)
return encoded
"""


class RedisExactCache:
    def __init__(self, url: str, timeout: float = 1, prefix: str = "cachewise:v1"):
        self.timeout = timeout
        self.prefix = prefix
        self.client = Redis.from_url(
            url,
            decode_responses=True,
            protocol=2,  # Keep FT.SEARCH wire responses stable across redis-py versions.
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            retry=Retry(NoBackoff(), 0),
        )

    async def _execute(self, operation, *args, **kwargs):
        try:
            async with asyncio.timeout(self.timeout):
                return await operation(*args, **kwargs)
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            raise CacheError("cache_unavailable") from exc

    async def _versions(self, scope: str) -> Namespace:
        raw = await self._execute(
            self.client.eval, NAMESPACE_SCRIPT, 1, f"{self.prefix}:namespace", str(uuid4()), scope
        )
        try:
            return Namespace.model_validate_json(raw)
        except ValueError as exc:
            raise CacheError("cache_namespace_invalid") from exc

    async def namespace(self) -> Namespace:
        return await self._versions("")

    async def invalidate(self, scope: Scope) -> Namespace:
        return await self._versions(scope)

    async def get(self, identity: str) -> CacheEntry | None:
        raw = await self._execute(self.client.get, f"{self.prefix}:exact:{identity}")
        if raw is None:
            return None
        try:
            entry = CacheEntry.model_validate_json(raw)
        except ValueError as exc:
            raise CacheError("cache_entry_invalid") from exc
        if entry.identity != identity or not entry.answer.strip():
            raise CacheError("cache_entry_invalid")
        if entry.expires_at <= time.time():
            return None
        return entry

    async def put(self, entry: CacheEntry, ttl: int) -> None:
        # NX avoids extending the first writer's TTL during concurrent misses.
        await self._execute(
            self.client.set,
            f"{self.prefix}:exact:{entry.identity}",
            entry.model_dump_json(),
            ex=ttl,
            nx=True,
        )

    async def ready(self) -> bool:
        try:
            return bool(await self._execute(self.client.ping))
        except CacheError:
            return False

    async def close(self) -> None:
        await self.client.aclose()
