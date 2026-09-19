import csv
import json
from collections import Counter

import httpx
import pytest

from cachewise.config import Settings
from cachewise.dataset import dataset_hash, make_dataset
from cachewise.models import Usage
from cachewise.replay import engine_snapshot, run_replay


def test_dataset_reproducibility_split_and_repetition():
    rows = make_dataset()
    assert len(rows) == 1000
    assert Counter(row.split for row in rows) == {"tuning": 300, "evaluation": 700}
    assert len({row.request.model_dump_json() for row in rows}) == 400
    families = [
        {row.family for row in rows if row.split == split} for split in ("tuning", "evaluation")
    ]
    assert not families[0] & families[1]
    assert dataset_hash(rows) == dataset_hash(make_dataset())
    assert dataset_hash(rows) != dataset_hash(make_dataset(43))
    assert len({row.id for row in rows}) == 1000
    assert any(row.expected_intent == "personalized" for row in rows)
    assert {row.context["tier"] for row in rows} == {"standard", "premium"}
    assert {row.context["region"] for row in rows} == {"US", "EU"}


async def test_replay_sequential_exports_unknown_usage_and_errors(tmp_path):
    config = Settings(_env_file=None)
    calls = []

    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(
                200, json={"generation_provider": "ready", "generation_model": "cachewise-model"}
            )
        if request.url.path == "/metrics":
            if request.url.port == 8001:
                return httpx.Response(200, text='vllm:prefix_cache_hits_total{model="m"} 12\n')
            return httpx.Response(
                200,
                json={
                    "configuration": config.public_config(),
                    "generation_calls": len(calls),
                },
            )
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            return httpx.Response(502, json={"error": "upstream_http_500"})
        return httpx.Response(
            200,
            json={
                "answer": "Test answer",
                "request_id": str(len(calls)),
                "cache_outcome": "bypass",
                "usage": Usage(prompt_tokens=20).model_dump(),
                "model": "cachewise-model",
                "finish_reason": "stop",
                "system_prompt_hash": "abc",
                "serialized_prompt_hash": "def",
                "policy_version": 1,
                "catalogue_version": 1,
            },
        )

    rows = make_dataset()[:3]
    output = tmp_path / "baseline"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_replay(rows, output, config, warmup_count=0, client=client)
    assert calls == [row.request.model_dump() for row in rows]
    assert result["successes"] == 2 and result["failures"] == 1
    assert result["generation_calls"] == 3
    assert result["prompt_tokens"] == 40
    assert result["completion_tokens"] is None
    assert result["p95_latency_ms"] > 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_errors"
    assert manifest["engine_before"]["prefix_series"][0]["value"] == 12
    assert len((output / "requests.jsonl").read_text().splitlines()) == 3
    with (output / "aggregate.csv").open() as stream:
        aggregate = next(csv.DictReader(stream))
    assert aggregate["generation_dollar_estimate"] == "unavailable"
    with pytest.raises(FileExistsError):
        await run_replay(rows, output, config)


async def test_preflight_preserves_diagnostics(tmp_path):
    transport = httpx.MockTransport(
        lambda r: httpx.Response(503, json={"generation_provider": "unavailable"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="preflight failed"):
            await run_replay(
                make_dataset()[:1], tmp_path / "failed", Settings(_env_file=None), client=client
            )
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "preflight_failed"
    assert not (tmp_path / "failed/requests.jsonl").exists()


async def test_missing_engine_metrics_are_unavailable(tmp_path):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="# no prefix metrics\n"))
    async with httpx.AsyncClient(transport=transport) as client:
        snapshot = await engine_snapshot(client, "http://server/metrics", tmp_path / "metrics.prom")
    assert snapshot["availability"] == "unavailable"


@pytest.mark.parametrize("runner_enabled", [True, False])
async def test_baseline_rejects_caching_before_warmup(tmp_path, runner_enabled):
    runner = Settings(_env_file=None, cache_mode="exact" if runner_enabled else "disabled")
    api = Settings(_env_file=None, cache_mode="exact")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"generation_provider": "ready"})
        return httpx.Response(200, json={"configuration": api.public_config()})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="preflight failed"):
            await run_replay(make_dataset()[:1], tmp_path / "reject", runner, client=client)
    assert "/chat" not in calls
    assert not (tmp_path / "reject/warmup.jsonl").exists()
