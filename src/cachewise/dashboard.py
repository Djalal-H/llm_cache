"""Read-only local dashboard for operational metrics and saved evaluation evidence."""

import json
import os
from pathlib import Path

import httpx
import streamlit as st


def display(value, suffix="") -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}" if isinstance(value, int) else str(value)


def rate(hits, requests):
    return 100 * hits / requests if hits is not None and requests else None


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def load_metrics(api_url: str) -> dict:
    response = httpx.get(f"{api_url.rstrip('/')}/metrics", timeout=5)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Expected metrics object")
    return value


def cards(items):
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value)


def live_metrics(api_url):
    st.header("Live application")
    try:
        metrics = load_metrics(api_url)
    except (httpx.HTTPError, ValueError):
        st.warning(
            "API metrics unavailable. Check the API address and refresh; "
            "saved runs remain available."
        )
        return
    st.caption(metrics.get("metrics_scope", "This API process since startup"))
    requests = metrics.get("requests")
    cards(
        [
            ("Requests", display(requests)),
            ("Exact hit rate", display(rate(metrics.get("exact_hits"), requests), "%")),
            ("Semantic hit rate", display(rate(metrics.get("semantic_hits"), requests), "%")),
            ("Avoided generation calls", display(metrics.get("avoided_generation_calls"))),
        ]
    )
    latency = metrics.get("latency_ms") or {}
    cards(
        [
            ("p50 latency", display(latency.get("p50"), " ms")),
            ("p95 latency", display(latency.get("p95"), " ms")),
            ("Failed requests", display(metrics.get("failures"))),
            ("Cache errors", display(metrics.get("cache_errors"))),
        ]
    )
    st.caption(
        "Hit rates include all requests. Latency covers the last 10,000 requests, including errors."
    )
    st.write("Cache mode:", metrics.get("application_cache", "Unavailable"))
    st.write("Active policy/catalogue versions", metrics.get("active_versions"))
    st.write("Cache namespace", metrics.get("cache_namespace"))
    st.subheader("Token usage")
    usage = metrics.get("usage") or {}
    st.dataframe(
        [
            {
                "Usage": name,
                "Known tokens": entry.get("known_total"),
                "Observed generations": entry.get("observed_requests"),
                "Generations without usage": entry.get("unavailable_requests"),
            }
            for name, entry in usage.items()
            if name != "prefix_cached_tokens"
        ],
        hide_index=True,
    )
    st.write("Embedding usage", metrics.get("embedding_usage"))
    st.subheader("vLLM prefix reuse")
    prefix = usage.get("prefix_cached_tokens") or {}
    st.metric("Engine-reported reused prompt tokens", display(prefix.get("known_total")))
    st.caption(
        "Per-generation response usage; unavailable observations are not zero. "
        "Prefix reuse is separate from application response-cache hits."
    )
    st.write("Prefix usage observations", prefix)
    with st.expander("API configuration and raw metrics"):
        st.json(metrics)


def saved_evidence(root: Path):
    st.header("Saved evaluation evidence")
    st.caption(
        "Select artifacts explicitly. "
        "Live counters and saved runs have different measurement scopes."
    )
    for filename, label, renderer in [
        ("manifest.json", "Replay run", replay_view),
        ("comparison.csv", "Validated comparison", comparison_view),
        ("summary.json", "Semantic judgments", judgment_view),
    ]:
        paths = sorted(p for p in root.rglob(filename) if p.is_file()) if root.is_dir() else []
        selected = st.selectbox(
            label,
            [None, *paths],
            format_func=lambda p: "Select evidence" if p is None else str(p.relative_to(root)),
        )
        if selected is not None:
            try:
                renderer(selected)
            except (OSError, ValueError, KeyError, TypeError):
                st.warning(f"Could not read {label.lower()}; it may be incomplete or invalid.")
    st.info(
        "Outstanding evidence: automated tuning/threshold freeze, held-out quality gate, "
        "50-case human review, and the dedicated prefix enabled-versus-disabled experiment. "
        "Saved replay or judge results alone do not satisfy these gates."
    )


def replay_view(path):
    manifest = read_object(path)
    st.write("Run status:", manifest.get("status", "incomplete"))
    st.write("Traffic splits:", manifest.get("splits", []))
    aggregate = manifest.get("aggregate")
    if not aggregate:
        st.warning("This run has no completed aggregate measurements.")
        return
    st.dataframe([aggregate], hide_index=True)
    st.subheader("Recorded costs")
    cards(
        [
            ("Generation estimate ($)", display(manifest.get("generation_dollar_estimate"))),
            ("Embedding cost ($)", display(manifest.get("embedding_cost"))),
            ("Judge overhead ($)", display(manifest.get("judge_cost"))),
        ]
    )
    st.caption(
        "Net monetary savings unavailable: no generation serving-cost model is implemented. "
        "Paid endpoint prices are not yet accounted for. Judge costs are evaluation overhead."
    )
    st.write(
        "Infrastructure cost assumptions:", manifest.get("infrastructure_costs", "Unavailable")
    )
    with st.expander("Engine prefix evidence and reproducibility"):
        st.caption(
            "Engine snapshots are aggregate and may include other traffic. "
            "They do not establish an enabled-versus-disabled latency benefit."
        )
        st.json(manifest)
    st.download_button(
        "Download replay manifest", path.read_bytes(), "manifest.json", "application/json"
    )


def comparison_view(path):
    import pandas as pd

    frame = pd.read_csv(path, na_values=["unavailable"])
    if "application_cache" not in frame or set(frame.application_cache) != {
        "disabled",
        "exact",
        "semantic",
    }:
        raise ValueError("Expected three cache modes")
    st.dataframe(frame, hide_index=True)
    for columns, title in [
        (["exact_hit_rate", "semantic_hit_rate"], "Application hit rates"),
        (["p50_latency_ms", "p95_latency_ms"], "End-to-end latency (ms)"),
        (["prompt_tokens", "completion_tokens", "embedding_total_tokens"], "Reported token usage"),
    ]:
        if all(column in frame for column in columns):
            st.write(title)
            st.bar_chart(frame.set_index("application_cache")[columns])
    st.caption(
        "Deltas are cached minus baseline. Negative token/latency deltas indicate reductions. "
        "Missing measurements remain unavailable. Prefix warm-state equivalence is unverified."
    )
    st.download_button("Download comparison", path.read_bytes(), "comparison.csv", "text/csv")


def judgment_view(path):
    summary = read_object(path)
    total, failures = summary["semantic_hits"], summary["false_hits"]
    st.write("Evaluated splits:", summary.get("splits", []))
    st.write("Dataset SHA-256:", summary.get("dataset_sha256", "Unavailable"))
    cards(
        [
            ("False semantic hits", f"{failures}/{total}"),
            ("Observed false-hit rate", display(rate(failures, total), "%")),
            ("Observed <1% gate", summary.get("observed_gate", "insufficient evidence")),
        ]
    )
    interval = summary.get("wilson_95_interval")
    st.write(
        "95% Wilson interval:",
        f"{interval[0]:.2%}–{interval[1]:.2%}" if interval else "Unavailable",
    )
    st.caption(summary.get("interval_caveat", "Repeated requests are correlated."))
    st.warning("An observed pass does not establish a population false-hit rate below 1%.")
    st.write("Manual review:", summary.get("manual_review", "pending"))
    st.json(summary)


def main():
    st.set_page_config(page_title="Cachewise", page_icon="🗃️", layout="wide")
    st.title("Cachewise")
    st.caption("Support response caching · operational metrics and measured replay evidence")
    api_url = st.sidebar.text_input(
        "API URL", os.getenv("CACHEWISE_DASHBOARD_API_URL", "http://127.0.0.1:8000")
    )
    root = Path(os.getenv("CACHEWISE_DASHBOARD_ARTIFACTS_DIR", "artifacts"))
    st.sidebar.caption(f"Evidence directory: {root}")
    st.sidebar.button("Refresh")
    live_metrics(api_url)
    saved_evidence(root)


if __name__ == "__main__":
    main()
