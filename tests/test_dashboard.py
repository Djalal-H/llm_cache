import json
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from cachewise import dashboard
from cachewise.metrics import Metrics
from cachewise.models import Usage

APP = Path(dashboard.__file__)


def setup_app(monkeypatch, tmp_path, metrics=None):
    monkeypatch.setenv("CACHEWISE_DASHBOARD_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            200, json=metrics or Metrics().snapshot(), request=httpx.Request("GET", args[0])
        ),
    )
    return AppTest.from_file(str(APP), default_timeout=15)


def test_empty_metrics_and_missing_artifacts(monkeypatch, tmp_path):
    app = setup_app(monkeypatch, tmp_path).run()
    assert not app.exception
    values = {m.label: m.value for m in app.metric}
    assert values["Exact hit rate"] == "Unavailable"
    assert values["Engine-reported reused prompt tokens"] == "Unavailable"
    assert values["Avoided generation calls"] == "0"


def test_live_metrics_preserve_prefix_scope(monkeypatch, tmp_path):
    metrics = Metrics("semantic")
    metrics.requests = 4
    metrics.exact_hits = 1
    metrics.semantic_hits = 1
    metrics.record_usage(Usage(prompt_tokens=100, prefix_cached_tokens=64))
    metrics.record_usage(Usage())
    app = setup_app(monkeypatch, tmp_path, metrics.snapshot()).run()
    assert not app.exception
    values = {m.label: m.value for m in app.metric}
    assert values["Exact hit rate"] == "25.00%"
    assert values["Semantic hit rate"] == "25.00%"
    assert values["Avoided generation calls"] == "2"
    assert values["Engine-reported reused prompt tokens"] == "64"


def test_offline_api_still_displays_judge_uncertainty(monkeypatch, tmp_path):
    app = setup_app(monkeypatch, tmp_path)

    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline)
    path = tmp_path / "judge" / "summary.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "semantic_hits": 10,
                "false_hits": 0,
                "wilson_95_interval": [0, 0.2775],
                "observed_gate": "passed",
                "splits": ["tuning"],
                "manual_review": "pending",
            }
        )
    )
    app.run()
    app.selectbox[2].select(path).run()
    assert not app.exception
    assert any("API metrics unavailable" in w.value for w in app.warning)
    assert any("population" in w.value for w in app.warning)
    values = {m.label: m.value for m in app.metric}
    assert values["False semantic hits"] == "0/10"


def test_saved_replay_comparison_and_invalid_file(monkeypatch, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "aggregate": {"requests": 10}}))
    comparison = run / "comparison.csv"
    comparison.write_text(
        "application_cache,p50_latency_ms,p95_latency_ms,exact_hit_rate,semantic_hit_rate\n"
        "disabled,100,200,0,0\nexact,50,200,0.5,0\nsemantic,40,180,0.5,0.1\n"
    )
    app = setup_app(monkeypatch, tmp_path).run()
    app.selectbox[0].select(manifest).run()
    app.selectbox[1].select(comparison).run()
    assert not app.exception
    values = {m.label: m.value for m in app.metric}
    assert values["Generation estimate ($)"] == "Unavailable"
    assert len(app.get("download_button")) == 2
    manifest.write_text("{broken")
    app.run()
    assert not app.exception
    assert any("incomplete or invalid" in w.value for w in app.warning)
