"""Evaluate saved replay evidence without rerunning generation or changing cache state."""

import asyncio
import csv
import json
import math
from pathlib import Path

import httpx

from cachewise.config import Settings
from cachewise.providers import JudgeProvider, OpenAIJudge, ProviderError


def wilson_interval(failures: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = failures / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - margin), min(1, center + margin)]


async def judge_replay(
    run: Path, output: Path, settings: Settings, judge: JudgeProvider | None = None
) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("judging requires a completed replay")
    if judge is None and not settings.judge_base_url:
        raise ValueError("configure CACHEWISE_JUDGE_BASE_URL and CACHEWISE_JUDGE_MODEL")
    rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
    hits = [
        r
        for r in rows
        if r["status"] == "success" and r["response"]["cache_outcome"] == "semantic_hit"
    ]
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    judgments = []
    async with httpx.AsyncClient() as client:
        provider = judge or OpenAIJudge(
            client,
            settings.judge_base_url,
            settings.judge_model,
            settings.judge_api_key.get_secret_value(),
            settings.judge_timeout_seconds,
        )
        with (output / "judgments.jsonl").open("w") as stream:
            for index, row in enumerate(hits):
                result = {
                    "dataset_id": row["dataset_request"]["id"],
                    "passed": False,
                    "error": None,
                    "usage": None,
                    "question": row["dataset_request"]["request"]["question"],
                    "answer": row["response"]["answer"],
                    "requirements": row.get("judge_requirements"),
                    "similarity": row["response"].get("semantic_similarity"),
                }
                if not row.get("judge_context_verified") or not result["requirements"]:
                    result["error"] = "unverified_policy_context"
                else:
                    try:
                        judgment = await provider.judge(
                            result["question"],
                            result["answer"],
                            json.dumps(result["requirements"], ensure_ascii=False),
                        )
                        result["passed"] = judgment.passed is True
                        result["error"] = "uncertain" if judgment.passed is None else None
                        result["usage"] = judgment.usage.model_dump()
                        result["model"] = judgment.model
                    except ProviderError as exc:
                        result["error"] = exc.code
                judgments.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Judged {index + 1}/{len(hits)} semantic hits", flush=True)
    failures = sum(not r["passed"] for r in judgments)
    total = len(judgments)
    summary = {
        "dataset_sha256": manifest["dataset_sha256"],
        "splits": manifest["splits"],
        "threshold": manifest["configuration"].get("semantic_threshold"),
        "semantic_hits": total,
        "false_hits": failures,
        "false_hit_rate": failures / total if total else None,
        "wilson_95_interval": wilson_interval(failures, total),
        "interval_caveat": "Descriptive hit-level interval; repeated requests are correlated.",
        "observed_gate": "insufficient evidence"
        if not total
        else "passed"
        if failures / total < 0.01
        else "failed",
        "judge_model": settings.judge_model,
        "judge_cost": None,
        "manual_review": "pending",
        "replay_failures": manifest["aggregate"]["failures"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Prioritize failures, then the lowest-similarity hits for human review.
    review = sorted(judgments, key=lambda r: (r["passed"], r["similarity"] or 0))[:50]
    with (output / "manual-review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dataset_id",
                "question",
                "answer",
                "requirements",
                "judge_passed",
                "human_passed",
                "notes",
            ],
        )
        writer.writeheader()
        for row in review:
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "requirements": json.dumps(row["requirements"]),
                    "judge_passed": row["passed"],
                    "human_passed": "",
                    "notes": "",
                }
            )
    (output / "report.md").write_text(
        "# Semantic reuse evaluation\n\n"
        f"Observed false hits: {failures}/{total}. Gate: {summary['observed_gate']}.\n\n"
        f"95% Wilson interval: {summary['wilson_95_interval']}. "
        "Repeated traffic is correlated; this is not a population guarantee.\n\n"
        f"Manual review: pending ({len(review)} cases exported; target 50). "
        "Threshold selection, held-out evaluation and prefix experiment remain separate gates.\n\n"
        "Judge costs and self-hosted generation dollar savings: unavailable.\n"
    )
    return summary


def compare_replays(runs: list[Path], output: Path) -> list[dict]:
    manifests = [json.loads((run / "manifest.json").read_text()) for run in runs]
    if len(manifests) != 3 or {m["configuration"]["application_cache"] for m in manifests} != {
        "disabled",
        "exact",
        "semantic",
    }:
        raise ValueError("provide exactly one disabled, exact and semantic replay")
    baseline = next(m for m in manifests if m["configuration"]["application_cache"] == "disabled")
    for manifest in manifests:
        if manifest["status"] != "completed":
            raise ValueError("comparison requires successful complete replays")
        for key in ("dataset_sha256", "warmup_count", "serving"):
            if manifest[key] != baseline[key]:
                raise ValueError(f"comparison mismatch: {key}")
        for key in ("generation_model", "generation_model_revision", "temperature", "max_tokens"):
            if manifest["configuration"][key] != baseline["configuration"][key]:
                raise ValueError(f"comparison mismatch: {key}")
    prompt_sequences = []
    for run in runs:
        records = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
        if len(records) != baseline["aggregate"]["requests"]:
            raise ValueError("comparison has incomplete request evidence")
        prompt_sequences.append(
            [(r["dataset_request"]["id"], r["response"]["serialized_prompt_hash"]) for r in records]
        )
    if any(sequence != prompt_sequences[0] for sequence in prompt_sequences[1:]):
        raise ValueError("comparison mismatch: actual request prompts")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for manifest in manifests:
        aggregate = manifest["aggregate"]
        row = dict(aggregate)
        for key in (
            "generation_calls",
            "prompt_tokens",
            "completion_tokens",
            "p50_latency_ms",
            "p95_latency_ms",
        ):
            value, reference = aggregate.get(key), baseline["aggregate"].get(key)
            row[f"{key}_delta"] = (
                value - reference if value is not None and reference is not None else None
            )
        row["exact_hit_rate"] = aggregate["exact_hits"] / aggregate["requests"]
        row["semantic_hit_rate"] = aggregate["semantic_hits"] / aggregate["requests"]
        rows.append(row)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.md").write_text(
        "# Application cache comparison\n\n"
        "See comparison.csv. Deltas are cached minus baseline; negative latency/token deltas "
        "indicate reductions. Hit-rate denominators include all replay requests.\n\n"
        "Prefix warm-state equivalence: insufficient evidence (manual fresh-server procedure). "
        "Keep prefix caching enabled across these runs. Shared-server traffic can contaminate "
        "aggregate metrics. Semantic quality and manual review are separate gates.\n\n"
        "Self-hosted dollar savings: unavailable. Paid endpoint costs: unavailable.\n"
    )
    (output / "manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    return rows
