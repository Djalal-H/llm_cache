import json

import httpx
import pytest

from cachewise.config import Settings
from cachewise.dataset import make_dataset
from cachewise.evaluation import compare_replays, judge_replay, wilson_interval
from cachewise.models import JudgeResult, Usage
from cachewise.providers import ProviderError
from cachewise.replay import run_replay


async def test_cached_replay_resets_after_warmup_and_counts_hits(tmp_path):
    settings = Settings(_env_file=None, cache_mode="exact", admin_token="test-secret")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.port == 8001:
            return httpx.Response(404)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "generation_provider": "ready",
                    "generation_model": "cachewise-model",
                    "response_cache": "ready",
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(
                200, json={"configuration": settings.public_config(), "generation_calls": 1}
            )
        if request.url.path == "/admin/invalidate":
            assert request.headers["authorization"] == "Bearer test-secret"
            return httpx.Response(200, json={"cache_namespace": {"policy": 2}})
        return httpx.Response(
            200,
            json={
                "answer": "answer",
                "request_id": "test",
                "cache_outcome": "exact_hit",
                "usage": Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0).model_dump(),
                "model": "cachewise-model",
                "finish_reason": "stop",
                "system_prompt_hash": "a",
                "serialized_prompt_hash": "b",
                "policy_version": 1,
                "catalogue_version": 1,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_replay(
            make_dataset()[:2],
            tmp_path / "run",
            settings,
            warmup_count=1,
            client=client,
            allow_cache=True,
        )
    assert calls.index("/chat") < calls.index("/admin/invalidate")
    assert calls[calls.index("/admin/invalidate") + 1 :].count("/chat") == 2
    assert result["exact_hits"] == result["avoided_generation_calls"] == 2
    assert result["application_cache"] == "exact"
    assert "test-secret" not in (tmp_path / "run/manifest.json").read_text()


async def test_judge_all_hits_fail_closed_and_preserve_context(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset_sha256": "hash",
                "splits": ["tuning"],
                "configuration": {"semantic_threshold": 0.9},
                "aggregate": {"failures": 0},
            }
        )
    )
    rows = [
        {
            "status": "success",
            "dataset_request": {"id": str(i), "request": {"question": "question"}},
            "response": {"cache_outcome": "semantic_hit", "answer": "answer"},
            "judge_context_verified": i != 3,
            "judge_requirements": {"expected_answer": "policy"},
        }
        for i in range(4)
    ]
    (run / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    class Judge:
        calls = 0

        async def judge(self, question, answer, requirements):
            assert json.loads(requirements)["expected_answer"] == "policy"
            self.calls += 1
            if self.calls == 3:
                raise ProviderError("upstream_timeout")
            return JudgeResult(passed=True if self.calls == 1 else None)

    judge = Judge()
    result = await judge_replay(run, tmp_path / "judged", Settings(_env_file=None), judge)
    assert result["false_hits"] == 3 and result["semantic_hits"] == 4
    assert judge.calls == 3
    assert result["observed_gate"] == "failed"
    assert (tmp_path / "judged/manual-review.csv").exists()
    with pytest.raises(FileExistsError):
        await judge_replay(run, tmp_path / "judged", Settings(_env_file=None), judge)


def test_interval_retains_uncertainty_with_zero_failures():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 50)[1] > 0.01
    assert wilson_interval(50, 50)[1] == pytest.approx(1)


def test_comparison_rejects_different_traffic(tmp_path):
    runs = []
    for mode in ("disabled", "exact", "semantic"):
        run = tmp_path / mode
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "application_cache": mode,
                        "generation_model": "m",
                        "generation_model_revision": None,
                        "temperature": 0,
                        "max_tokens": 20,
                    },
                    "dataset_sha256": mode,
                    "warmup_count": 2,
                    "serving": None,
                    "status": "completed",
                }
            )
        )
        runs.append(run)
    with pytest.raises(ValueError, match="dataset_sha256"):
        compare_replays(runs, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_comparison_exports_deltas_and_rejects_prompt_changes(tmp_path):
    runs = []
    for mode, tokens in (("disabled", 100), ("exact", 70), ("semantic", 50)):
        run = tmp_path / mode
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "application_cache": mode,
                        "generation_model": "m",
                        "generation_model_revision": None,
                        "temperature": 0,
                        "max_tokens": 20,
                    },
                    "dataset_sha256": "same",
                    "warmup_count": 2,
                    "serving": None,
                    "status": "completed",
                    "aggregate": {
                        "application_cache": mode,
                        "requests": 1,
                        "prompt_tokens": tokens,
                        "exact_hits": 0,
                        "semantic_hits": 0,
                    },
                }
            )
        )
        (run / "requests.jsonl").write_text(
            json.dumps(
                {
                    "dataset_request": {"id": "one"},
                    "response": {"serialized_prompt_hash": "same"},
                }
            )
            + "\n"
        )
        runs.append(run)
    rows = compare_replays(runs, tmp_path / "comparison")
    assert rows[2]["prompt_tokens_delta"] == -50
    assert (tmp_path / "comparison/comparison.csv").exists()
    (runs[2] / "requests.jsonl").write_text(
        json.dumps(
            {
                "dataset_request": {"id": "one"},
                "response": {"serialized_prompt_hash": "changed"},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="actual request prompts"):
        compare_replays(runs, tmp_path / "rejected")
