"""User-run live exact-cache smoke test. Never part of the automated test suite."""

import argparse
import json
import time

import httpx

from cachewise.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    settings = Settings()
    token = settings.admin_token.get_secret_value()
    if not token:
        raise SystemExit("Set CACHEWISE_ADMIN_TOKEN to the same token used by the API.")
    with httpx.Client(
        base_url=args.api_url, timeout=settings.generation_timeout_seconds + 15
    ) as client:

        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        metrics = request("GET", "/metrics")
        if metrics["configuration"]["application_cache"] != "exact":
            raise SystemExit("Restart the API with CACHEWISE_CACHE_MODE=exact first.")
        headers = {"Authorization": f"Bearer {token}"}

        def invalidate(scope):
            result = request("POST", "/admin/invalidate", headers=headers, json={"scope": scope})
            print(json.dumps({"action": "invalidate", **result}, indent=2), flush=True)

        # Make the test repeatable even if the question was previously cached.
        invalidate("all")
        before = request("GET", "/metrics")
        body = {"customer_id": "demo-alice", "question": "How much does shipping cost?"}
        responses = []
        for index, expected in enumerate(("miss", "exact_hit", "miss"), start=1):
            if index == 3:
                invalidate("policy")
            started = time.perf_counter()
            result = request("POST", "/chat", json=body)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            print(
                json.dumps({"step": index, "latency_ms": elapsed, **result}, indent=2), flush=True
            )
            if result["cache_outcome"] != expected:
                raise SystemExit(f"Expected {expected}; received {result['cache_outcome']}.")
            responses.append(result)
        after = request("GET", "/metrics")
        delta = {
            name: after[name] - before[name]
            for name in ("generation_calls", "exact_hits", "cache_misses", "cache_errors")
        }
        print(json.dumps({"metrics_delta": delta}, indent=2), flush=True)
        if delta != {"generation_calls": 2, "exact_hits": 1, "cache_misses": 2, "cache_errors": 0}:
            raise SystemExit("Unexpected metrics; check cache errors and concurrent API traffic.")
        if responses[0]["answer"] != responses[1]["answer"]:
            raise SystemExit("The exact hit did not preserve the original answer.")
        if len({result["request_id"] for result in responses}) != 3:
            raise SystemExit("Responses did not receive distinct request identifiers.")
        print("PASS: miss -> exact_hit -> invalidate -> miss; two generation calls.")


if __name__ == "__main__":
    main()
