import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from cachewise.assistant import Assistant
from cachewise.cache import CacheError, ExactCache, RedisExactCache
from cachewise.chat import ChatService
from cachewise.config import Settings
from cachewise.fixtures import FixtureService
from cachewise.metrics import Metrics
from cachewise.models import ChatRequest, ChatResponse, InvalidateRequest
from cachewise.providers import (
    GenerationProvider,
    OpenAIEmbedding,
    OpenAIGeneration,
    OpenAIJudge,
    ProviderError,
)


def get_assistant(request: Request) -> Assistant:
    return request.app.state.assistant


def create_app(
    settings: Settings | None = None,
    generation: GenerationProvider | None = None,
    cache: ExactCache | None = None,
) -> FastAPI:
    config = settings or Settings()
    fixtures = FixtureService(config.fixture_path)
    metrics = Metrics(config.cache_mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        fixtures.snapshot()  # Validate fictional data before accepting requests.
        async with httpx.AsyncClient() as client:
            provider = generation or OpenAIGeneration(
                client,
                config.generation_base_url,
                config.generation_model,
                config.generation_api_key.get_secret_value(),
                config.generation_timeout_seconds,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            app.state.assistant = Assistant(fixtures, provider)
            active_cache = cache
            if active_cache is None and config.cache_mode == "exact":
                active_cache = RedisExactCache(
                    config.redis_url.get_secret_value(), config.cache_timeout_seconds
                )
            app.state.cache = active_cache
            app.state.chat_service = ChatService(app.state.assistant, config, active_cache, metrics)
            app.state.embedding = (
                OpenAIEmbedding(
                    client,
                    config.embedding_base_url,
                    config.embedding_model,
                    config.embedding_api_key.get_secret_value(),
                    config.embedding_timeout_seconds,
                )
                if config.embedding_base_url
                else None
            )
            app.state.judge = (
                OpenAIJudge(
                    client,
                    config.judge_base_url,
                    config.judge_model,
                    config.judge_api_key.get_secret_value(),
                    config.judge_timeout_seconds,
                )
                if config.judge_base_url
                else None
            )
            try:
                yield
            finally:
                if active_cache is not None:
                    await active_cache.close()

    app = FastAPI(title="Cachewise", version="0.1.0", lifespan=lifespan)
    app.state.metrics = metrics
    app.state.settings = config

    @app.middleware("http")
    async def measure_chat(request: Request, call_next):
        if request.url.path != "/chat" or request.method != "POST":
            return await call_next(request)
        metrics.requests += 1
        request.state.request_id = str(uuid4())
        started = time.perf_counter()
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                metrics.failures += 1
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        except Exception:
            metrics.failures += 1
            raise
        finally:
            metrics.latencies.append((time.perf_counter() - started) * 1000)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest, request: Request, assistant: Annotated[Assistant, Depends(get_assistant)]
    ):
        try:
            prepared = assistant.prepare(body)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "unknown_customer",
                    "request_id": request.state.request_id,
                },
            )
        except (OSError, ValueError, ValidationError):
            return JSONResponse(
                status_code=503,
                content={
                    "error": "fixture_unavailable",
                    "request_id": request.state.request_id,
                },
            )
        try:
            response = await request.app.state.chat_service.answer(
                body, prepared, request.state.request_id
            )
        except ProviderError as exc:
            return JSONResponse(
                status_code=504 if exc.timed_out else 502,
                content={
                    "error": exc.code,
                    "request_id": request.state.request_id,
                    "system_prompt_hash": prepared.system_prompt_hash,
                    "serialized_prompt_hash": prepared.serialized_prompt_hash,
                    "policy_version": prepared.policy_version,
                    "catalogue_version": prepared.catalogue_version,
                },
            )
        return response

    @app.post("/admin/invalidate")
    async def invalidate(body: InvalidateRequest, request: Request):
        token = config.admin_token.get_secret_value()
        if not token:
            return JSONResponse(status_code=404, content={"error": "invalidation_disabled"})
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        active_cache = request.app.state.cache
        if config.cache_mode != "exact" or active_cache is None:
            return JSONResponse(status_code=503, content={"error": "cache_disabled"})
        try:
            namespace = await active_cache.invalidate(body.scope)
        except CacheError:
            metrics.cache_errors += 1
            return JSONResponse(status_code=503, content={"error": "cache_unavailable"})
        return {"scope": body.scope, "cache_namespace": namespace.model_dump()}

    @app.get("/health")
    async def health(assistant: Annotated[Assistant, Depends(get_assistant)]):
        ready = await assistant.generation.ready()
        try:
            fixtures.snapshot()
            fixture_ready = True
        except (OSError, ValueError):
            fixture_ready = False
        return JSONResponse(
            status_code=200 if ready and fixture_ready else 503,
            content={
                "application": "ready",
                "generation_provider": "ready" if ready else "unavailable",
                "fixtures": "ready" if fixture_ready else "unavailable",
                "generation_model": config.generation_model,
                "response_cache": (
                    "disabled"
                    if config.cache_mode == "disabled"
                    else "ready"
                    if await app.state.cache.ready()
                    else "unavailable"
                ),
            },
        )

    @app.get("/metrics")
    async def get_metrics():
        result = metrics.snapshot()
        result["configuration"] = config.public_config()
        try:
            snapshot = fixtures.snapshot()
            result["active_versions"] = {
                "policy": snapshot.policy_version,
                "catalogue": snapshot.catalogue_version,
            }
        except (OSError, ValueError):
            result["active_versions"] = None
        result["cache_namespace"] = None
        if config.cache_mode == "exact" and app.state.cache is not None:
            try:
                result["cache_namespace"] = (await app.state.cache.namespace()).model_dump()
            except CacheError:
                pass
        return result

    return app


app = create_app()
