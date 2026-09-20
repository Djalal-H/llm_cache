"""Filtered Redis cosine retrieval. Only generated answers become vector entries."""

import math
import re
import struct
import time
from dataclasses import replace
from typing import Protocol

from redis.exceptions import ResponseError

from cachewise.assistant import PreparedPrompt, canonical, digest
from cachewise.cache import CacheEntry, CacheError, Namespace, RedisExactCache
from cachewise.config import Settings


def embedding_namespace(config: Settings) -> str:
    return digest(
        canonical(
            {
                "endpoint": config.embedding_base_url,
                "model": config.embedding_model,
                "revision": config.embedding_model_revision,
                "dimension": config.embedding_dimension,
                "format": "float32-cosine-v1",
            }
        )
    )


def semantic_context(prompt: PreparedPrompt, config: Settings, namespace: Namespace) -> str:
    from cachewise.chat import exact_identity

    # Hash every actual context message, excluding only the variable question.
    context = replace(
        prompt,
        serialized_prompt_hash=digest(
            canonical([message.model_dump() for message in prompt.messages[:-1]])
        ),
    )
    return digest(
        canonical(
            {
                "identity": exact_identity(context, config, namespace),
                "guards": safety_guards(prompt.messages[-1].content),
            }
        )
    )


def safety_guards(question: str) -> list[str]:
    """Conservative prefilters for common high-similarity policy near-misses.

    These reduce coverage deliberately; they do not replace threshold evaluation.
    """
    words = set(re.findall(r"\w+", question.casefold()))
    groups = {
        "negation": {"not", "no", "without"},
        "us": {"us", "states"},
        "eu": {"eu", "european"},
        "custom": {"custom", "personalized", "engraved"},
        "defective": {"defective", "faulty"},
        "used": {"used"},
        "unused": {"unused"},
        "opened": {"opened"},
        "packaging": {"packaging"},
        "calendar": {"calendar"},
        "business": {"business"},
    }
    guards = [name for name, terms in groups.items() if words & terms]
    sensitive = {
        "canada",
        "japan",
        "brazil",
        "algeria",
        "international",
        "outside",
        "premium",
        "cash",
        "visa",
        "mastercard",
        "paypal",
        "bitcoin",
        "crypto",
        "cryptocurrency",
        "ethereum",
        "bank",
        "wire",
        "minimum",
    }
    guards.extend(words & sensitive)
    guards.extend(re.findall(r"\d+(?:[.,]\d+)?", question))
    return sorted(guards)


def vector_bytes(vector: list[float], dimension: int) -> bytes:
    if len(vector) != dimension or not all(math.isfinite(v) for v in vector):
        raise ValueError("invalid embedding dimension or values")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("invalid embedding norm")
    # Normalization avoids FLOAT32 overflow/underflow; cosine is unchanged.
    return struct.pack(f"<{dimension}f", *(value / norm for value in vector))


class SemanticHit(CacheEntry):
    context: str
    similarity: float


class SemanticCache(Protocol):
    async def search(self, context: str, vector: bytes, threshold: float) -> SemanticHit | None: ...
    async def put(self, context: str, vector: bytes, entry: CacheEntry, ttl: int) -> None: ...
    async def ready(self) -> bool: ...


PUT_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], 'context', ARGV[1], 'vector', ARGV[2],
           'entry', ARGV[3], 'expires', ARGV[4])
redis.call('PEXPIRE', KEYS[1], ARGV[5])
return 1
"""


class RedisSemanticCache:
    def __init__(self, cache: RedisExactCache, config: Settings):
        self.cache = cache
        self.dimension = config.embedding_dimension
        self.prefix = f"{cache.prefix}:semantic:{embedding_namespace(config)}:"
        self.index = self.prefix + "index"

    async def ensure_index(self) -> None:
        try:
            await self.cache._execute(
                self.cache.client.execute_command,
                "FT.CREATE",
                self.index,
                "ON",
                "HASH",
                "PREFIX",
                1,
                self.prefix + "entry:",
                "SCHEMA",
                "context",
                "TAG",
                "expires",
                "NUMERIC",
                "vector",
                "VECTOR",
                "FLAT",
                6,
                "TYPE",
                "FLOAT32",
                "DIM",
                self.dimension,
                "DISTANCE_METRIC",
                "COSINE",
            )
        except CacheError as exc:
            if (
                not isinstance(exc.__cause__, ResponseError)
                or str(exc.__cause__) != "Index already exists"
            ):
                raise

    async def ready(self) -> bool:
        try:
            await self.ensure_index()
            return True
        except CacheError:
            return False

    async def search(self, context: str, vector: bytes, threshold: float) -> SemanticHit | None:
        await self.ensure_index()
        if not re.fullmatch(r"[0-9a-f]{64}", context):
            raise CacheError("semantic_context_invalid")
        raw = await self.cache._execute(
            self.cache.client.execute_command,
            "FT.SEARCH",
            self.index,
            f"(@context:{{{context}}} @expires:[({time.time()} +inf])"
            "=>[KNN 1 @vector $v AS distance]",
            "PARAMS",
            2,
            "v",
            vector,
            "SORTBY",
            "distance",
            "RETURN",
            3,
            "entry",
            "context",
            "distance",
            "DIALECT",
            2,
        )
        try:
            if not raw[0]:
                return None
            fields = dict(zip(raw[2][::2], raw[2][1::2], strict=True))
            entry = CacheEntry.model_validate_json(fields["entry"])
            similarity = 1 - float(fields["distance"])
            if (
                fields["context"] != context
                or not math.isfinite(similarity)
                or not -1.000001 <= similarity <= 1.000001
                or entry.finish_reason != "stop"
                or not entry.answer.strip()
            ):
                raise ValueError("invalid semantic entry")
            if entry.expires_at <= time.time() or similarity < threshold:
                return None
            return SemanticHit(**entry.model_dump(), context=context, similarity=min(1, similarity))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise CacheError("semantic_entry_invalid") from exc

    async def put(self, context: str, vector: bytes, entry: CacheEntry, ttl: int) -> None:
        await self.ensure_index()
        remaining_ms = min(ttl * 1000, int((entry.expires_at - time.time()) * 1000))
        if remaining_ms <= 0:
            return
        await self.cache._execute(
            self.cache.client.eval,
            PUT_SCRIPT,
            1,
            self.prefix + "entry:" + entry.identity,
            context,
            vector,
            entry.model_dump_json(),
            entry.expires_at,
            remaining_ms,
        )
