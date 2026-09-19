import json

import httpx
import pytest

from cachewise.models import Message
from cachewise.providers import (
    OpenAIEmbedding,
    OpenAIGeneration,
    OpenAIJudge,
    ProviderError,
    parse_generation,
)


def completion(content="Real upstream answer", **extra):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], **extra}


def test_parse_usage_and_prefix_evidence():
    result = parse_generation(
        completion(
            model="live-model",
            usage={
                "prompt_tokens": 123,
                "completion_tokens": 20,
                "total_tokens": 143,
                "prompt_tokens_details": {"cached_tokens": 96},
            },
        )
    )
    assert result.usage.prefix_cached_tokens == 96
    assert result.usage.total_tokens == 143
    assert result.model == "live-model"


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 20}])
def test_missing_usage_is_unavailable(usage):
    result = parse_generation(completion(usage=usage))
    assert result.usage.completion_tokens is None
    assert result.usage.prefix_cached_tokens is None


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": []},
        completion(content=None),
        completion(content=""),
        completion(usage={"prompt_tokens": -1}),
        completion(usage={"prompt_tokens_details": [1]}),
    ],
)
def test_invalid_generation_response(data):
    with pytest.raises(ProviderError, match="upstream_invalid_response"):
        parse_generation(data)


async def test_generation_wire_request_and_readiness():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "cachewise-model"}]})
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIGeneration(client, "http://server/v1/", "cachewise-model", "test-key", 3)
        assert await provider.ready()
        result = await provider.generate([Message(role="user", content="Returns?")])
    body = json.loads(seen[-1].content)
    assert seen[-1].url.path == "/v1/chat/completions"
    assert seen[-1].headers["authorization"] == "Bearer test-key"
    assert body["temperature"] == 0 and body["max_tokens"] == 256
    assert result.answer == "Real upstream answer"


@pytest.mark.parametrize(
    "failure,code,timed_out",
    [
        (httpx.ReadTimeout("secret endpoint details"), "upstream_timeout", True),
        (httpx.ConnectError("secret endpoint details"), "upstream_unreachable", False),
        (503, "upstream_http_503", False),
        ("invalid-json", "upstream_invalid_response", False),
    ],
)
async def test_safe_upstream_errors(failure, code, timed_out):
    def handler(request):
        if isinstance(failure, Exception):
            raise failure
        if isinstance(failure, int):
            return httpx.Response(failure, text="sensitive upstream body")
        return httpx.Response(200, text="not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIGeneration(client, "http://server/v1", "model", "", 2)
        with pytest.raises(ProviderError) as caught:
            await provider.generate([])
        assert caught.value.code == code
        assert caught.value.timed_out == timed_out
        assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "vector,valid",
    [([0.1, 0.2], True), ([], False), ([True], False), (["0.1"], False), ([float("inf")], False)],
)
async def test_embedding_validation(vector, valid):
    transport = httpx.MockTransport(
        lambda r: (
            httpx.Response(
                200,
                json={
                    "data": [{"embedding": vector}],
                },
            )
            if valid or vector != [float("inf")]
            else httpx.Response(200, text='{"data":[{"embedding":[Infinity]}]}')
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIEmbedding(client, "http://server/v1", "embedding-model", "", 2)
        if valid:
            result = await provider.embed("Returns?")
            assert result.embedding == vector
            assert result.usage.total_tokens is None
        else:
            with pytest.raises(ProviderError):
                await provider.embed("Returns?")


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"pass":true}', True),
        ('{"pass":false}', False),
        ('{"pass":null}', None),
        ('{"pass":"yes"}', None),
        ("uncertain", None),
        ("[]", None),
    ],
)
async def test_judge_uncertain_and_unparseable_results(content, expected):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=completion(content)))
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAIJudge(client, "http://judge/v1", "judge-model", "", 2)
        result = await provider.judge("Returns?", "30 days", "Use policy")
        assert result.passed is expected
        assert result.usage.total_tokens is None


async def test_optional_providers_preserve_usage_for_later_accounting():
    def handler(request):
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1, 0.2]}],
                    "model": "embedding-model",
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            )
        return httpx.Response(
            200,
            json=completion(
                "uncertain",
                usage={
                    "prompt_tokens": 40,
                    "completion_tokens": 2,
                    "total_tokens": 42,
                },
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedding = await OpenAIEmbedding(
            client, "http://server/v1", "embedding-model", "", 2
        ).embed("Q")
        judgment = await OpenAIJudge(client, "http://server/v1", "judge-model", "", 2).judge(
            "Q", "A", "R"
        )
    assert embedding.usage.total_tokens == 8
    assert judgment.passed is None
    assert judgment.usage.total_tokens == 42
