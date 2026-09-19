import asyncio
import csv
import json
import platform
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from cachewise.config import Settings
from cachewise.dataset import DatasetRequest, dataset_hash
from cachewise.metrics import percentile
from cachewise.models import ChatResponse, Usage


async def engine_snapshot(client: httpx.AsyncClient, url: str, path: Path) -> dict:
    try:
        response = await client.get(url, timeout=5)
        response.raise_for_status()
        await asyncio.to_thread(path.write_text, response.text, encoding="utf-8")
        series = []
        for line in response.text.splitlines():
            if line.startswith("#") or "prefix_cache" not in line:
                continue
            match = re.fullmatch(r"(\S+(?:\{.*?\})?)\s+([\d.eE+\-]+)(?:\s+\d+)?", line)
            if match:
                series.append({"series": match[1], "value": float(match[2])})
        return {
            "availability": "available" if series else "unavailable",
            "scope": "aggregate engine snapshot; not evidence for an individual request",
            "prefix_series": series,
            "raw_file": path.name,
        }
    except (httpx.HTTPError, ValueError):
        return {"availability": "unavailable", "scope": "aggregate engine", "prefix_series": []}


def summarize(results: list[dict]) -> dict:
    successes = [row for row in results if row["status"] == "success"]
    usage = {}
    observations = {}
    for field in Usage.model_fields:
        values = [
            row["response"]["usage"][field]
            for row in successes
            if row["response"]["usage"][field] is not None
        ]
        usage[field] = sum(values) if values else None
        observations[f"{field}_observed_requests"] = len(values)
    latencies = [row["latency_ms"] for row in results]
    success_latencies = [row["latency_ms"] for row in successes]
    return {
        "requests": len(results),
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "success_p50_latency_ms": percentile(success_latencies, 0.5),
        "success_p95_latency_ms": percentile(success_latencies, 0.95),
        "truncated_answers": sum(row["response"]["finish_reason"] == "length" for row in successes),
        "application_cache": "disabled",
        "exact_hits": 0,
        "semantic_hits": 0,
        "avoided_generation_calls": 0,
        "generation_dollar_estimate": None,
        **usage,
        **observations,
    }


async def run_replay(
    rows: list[DatasetRequest],
    output: Path,
    settings: Settings,
    api_url: str = "http://127.0.0.1:8000",
    warmup_count: int = 2,
    serving_metadata: dict | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    if not rows:
        raise ValueError("replay needs at least one request")
    if warmup_count < 0:
        raise ValueError("warmup count must be nonnegative")
    # Refuse to overwrite evidence from a previous run.
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    if client is None:
        async with httpx.AsyncClient(timeout=settings.generation_timeout_seconds + 15) as owned:
            return await _replay(
                rows, output, settings, api_url, warmup_count, serving_metadata, owned
            )
    return await _replay(rows, output, settings, api_url, warmup_count, serving_metadata, client)


async def _replay(rows, output, settings, api_url, warmup_count, serving_metadata, client):
    api_url = api_url.rstrip("/")
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": dataset_hash(rows),
        "dataset_seeds": sorted({row.seed for row in rows}),
        "requests": len(rows),
        "splits": sorted({row.split for row in rows}),
        "concurrency": 1,
        "runner_python": platform.python_version(),
        "configuration": settings.public_config(),
        "serving": serving_metadata,
        "serving_metadata_source": "provided startup manifest"
        if serving_metadata
        else "unavailable",
        "warmup_count": warmup_count,
        "warmup_included_in_measurements": False,
        "prefix_cache_reset": "not performed; start a fresh vLLM server before this run",
        "interpretation": "uncached baseline only; no latency or savings comparison yet",
        "judge_cost": None,
        "embedding_cost": 0,
        "generation_dollar_estimate": None,
        "infrastructure_costs": "unavailable; no serving-cost model configured",
    }

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    try:
        if settings.cache_mode != "disabled":
            raise ValueError("baseline replay requires CACHEWISE_CACHE_MODE=disabled")
        health = await client.get(f"{api_url}/health", timeout=10)
        manifest["health"] = health.json()
        health.raise_for_status()
        if manifest["health"].get("generation_provider") != "ready":
            raise ValueError("generation provider is not ready")
        metrics = await client.get(f"{api_url}/metrics", timeout=10)
        metrics.raise_for_status()
        manifest["api_metrics_preflight"] = metrics.json()
        actual_config = manifest["api_metrics_preflight"].get("configuration")
        if (actual_config or {}).get("application_cache") != "disabled":
            raise ValueError("baseline replay requires application caching disabled on the API")
        if actual_config != settings.public_config():
            raise ValueError("runner and API generation configuration do not match")
        actual_model = manifest["health"].get("generation_model")
        if actual_model != settings.generation_model:
            raise ValueError("runner and API generation models do not match")
        # Warm up using the first requests, with every warm-up outcome preserved separately.
        with (output / "warmup.jsonl").open("w") as stream:
            for index in range(warmup_count):
                response = await client.post(
                    f"{api_url}/chat", json=rows[index % len(rows)].request.model_dump()
                )
                stream.write(
                    json.dumps({"status_code": response.status_code, "response": response.json()})
                    + "\n"
                )
                response.raise_for_status()
                ChatResponse.model_validate(response.json())
        metrics = await client.get(f"{api_url}/metrics", timeout=10)
        metrics.raise_for_status()
        manifest["api_metrics_before"] = metrics.json()
    except (httpx.HTTPError, ValueError) as exc:
        manifest["status"] = "preflight_failed"
        manifest["error"] = type(exc).__name__
        save_manifest()
        raise RuntimeError(f"Replay preflight failed; diagnostics saved in {output}") from exc

    manifest["engine_before"] = await engine_snapshot(
        client, settings.vllm_metrics_url, output / "engine-before.prom"
    )
    save_manifest()
    results = []
    with (output / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            record = {
                "dataset_request": row.model_dump(mode="json"),
                "configuration": settings.public_config(),
                "serving": serving_metadata,
                "response": None,
                "error": None,
                "started_at": datetime.now(UTC).isoformat(),
            }
            started = time.perf_counter()
            try:
                response = await client.post(f"{api_url}/chat", json=row.request.model_dump())
                record["http_status"] = response.status_code
                if response.is_success:
                    parsed = ChatResponse.model_validate(response.json())
                    if parsed.cache_outcome != "bypass":
                        raise ValueError("baseline requires application caching disabled")
                    record["response"] = parsed.model_dump()
                    record["status"] = "success"
                else:
                    record["status"] = "error"
                    try:
                        record["error"] = response.json()
                    except ValueError:
                        record["error"] = "non_json_api_error"
            except (httpx.HTTPError, ValueError) as exc:
                record["status"] = "error"
                record["error"] = type(exc).__name__
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(record)
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"Replayed {index + 1}/{len(rows)} requests", flush=True)

    aggregate = summarize(results)
    manifest["engine_after"] = await engine_snapshot(
        client, settings.vllm_metrics_url, output / "engine-after.prom"
    )
    try:
        response = await client.get(f"{api_url}/metrics", timeout=10)
        response.raise_for_status()
        manifest["api_metrics_after"] = response.json()
    except (httpx.HTTPError, ValueError):
        manifest["api_metrics_after"] = None
    before = manifest["api_metrics_before"].get("generation_calls")
    after = (manifest["api_metrics_after"] or {}).get("generation_calls")
    aggregate["generation_calls"] = (
        after - before
        if (isinstance(before, int) and isinstance(after, int) and after >= before)
        else None
    )
    with (output / "aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(
            {key: "unavailable" if value is None else value for key, value in aggregate.items()}
        )
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "completed" if aggregate["failures"] == 0 else "completed_with_errors"
    manifest["aggregate"] = aggregate
    save_manifest()
    return aggregate
