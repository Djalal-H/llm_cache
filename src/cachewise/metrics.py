import math
from collections import Counter, deque

from cachewise.models import Usage


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


class Metrics:
    def __init__(self, cache_mode: str = "disabled"):
        self.cache_mode = cache_mode
        self.exact_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        self.bypasses: Counter[str] = Counter()
        self.requests = 0
        self.failures = 0
        self.generation_calls = 0
        self.generation_successes = 0
        self.known_usage = dict.fromkeys(Usage.model_fields, 0)
        self.usage_observations = dict.fromkeys(Usage.model_fields, 0)
        self.latencies: deque[float] = deque(maxlen=10000)

    def record_usage(self, usage: Usage) -> None:
        self.generation_successes += 1
        for field, value in usage.model_dump().items():
            if value is not None:
                self.known_usage[field] += value
                self.usage_observations[field] += 1

    def snapshot(self) -> dict:
        values = list(self.latencies)
        return {
            "requests": self.requests,
            "failures": self.failures,
            "generation_calls": self.generation_calls,
            "generation_successes": self.generation_successes,
            "application_cache": self.cache_mode,
            "exact_hits": self.exact_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "bypasses": dict(self.bypasses),
            "avoided_generation_calls": self.exact_hits,
            "usage": {
                field: {
                    "known_total": total if self.usage_observations[field] else None,
                    "observed_requests": self.usage_observations[field],
                    "unavailable_requests": self.generation_successes
                    - self.usage_observations[field],
                }
                for field, total in self.known_usage.items()
            },
            "latency_ms": {
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "sample_count": len(values),
                "window": "last 10000 chat requests",
            },
            "generation_dollar_estimate": None,
            "prefix_evidence_source": "response.usage.prompt_tokens_details.cached_tokens",
            "metrics_scope": "this API process since startup",
        }
