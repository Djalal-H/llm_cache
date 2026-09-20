# Cachewise implementation progress

Last updated: 2026-09-20

## Project snapshot

**Phase 3 — semantic caching implemented and automatically verified.** The application still
defaults to `CACHEWISE_CACHE_MODE=disabled`; `exact` enables exact caching and `semantic` enables
exact-first cosine retrieval. Semantic mode requires an explicit embedding endpoint/model,
dimension and threshold. Live provider checks and semantic quality gates remain unverified.
Phase 4 evaluation is partially implemented: cached replay, saved-hit judging and comparison exist.
**Phase 5 — dashboard and handoff implemented**, with outstanding phase 4 evidence disclosed.

## Completed

- Foundation: provider-neutral generation/embedding/judge adapters, fixture-backed assistant,
  prompt hashes, API health/metrics, seeded 1,000-request dataset, baseline replay/export,
  native vLLM launcher and Docker API packaging.
- `src/cachewise/eligibility.py`: English FAQ intent rules and reviewed domain vocabulary with
  conservative bypasses for personal requests, unknown/mixed wording, stock, and history. It now
  marks a future TypeSafe AI Jev `Choice` integration seam; Jev is not installed or called.
- `src/cachewise/assistant.py`: eligible FAQ prompts contain policies, tier, region and language,
  without customer identifiers, names or orders; this also applies with caching disabled.
- `src/cachewise/cache.py` and `chat.py`: async Redis adapter and orchestration, complete prompt/model
  identity, 24-hour default TTL, explicit expiration rejection, no hit refresh, namespace counters,
  safe recovery of missing/corrupt namespace metadata, and cache-failure generation fallback.
- API outcomes `bypass`, `miss`, `exact_hit`; bearer-protected `/admin/invalidate`; separate Redis
  health; cache counters and active namespace reporting. Baseline replay rejects enabled caching
  during preflight, before warm-up.
- Redis 8.2 Compose service, validated environment settings, locked redis-py 8.1.0 dependency,
  automated cache/Redis tests, and startup/invalidation instructions in `README.md`.
- `scripts/smoke_exact_cache.py`: user-run live test printing responses, usage, latency, versions,
  and metric deltas; checks miss → exact_hit → invalidate → miss. Not executed by the agent.

- `src/cachewise/semantic.py`: Redis FLOAT32 FLAT cosine index per embedding identity/dimension,
  mandatory context/expiry prefilters, conservative near-match guards, finite nonzero vector
  validation, logical expiry rejection and atomic first-writer TTL. Redis wire protocol is pinned
  to RESP2 because redis-py 8 defaults to RESP3 and returns a different search response shape.
- `chat.py`, `api.py`, `config.py`, `models.py`, `metrics.py`: exact-first semantic orchestration,
  injected provider/cache seams, explicit validated settings, source identity/similarity and
  embedding usage in responses, separate embedding counters and semantic cache health.
- `tests/test_semantic.py` and expanded Redis integration tests cover approved paraphrases,
  adversarial guards, context/embedding identity isolation, outages, truncation, expiry,
  no promotion/refresh, invalidation during generation, and the real API sequence.
- `.env.example` and `README.md`: semantic configuration, manual live check, operating limits,
  and distinction between implemented retrieval and the future quality gate.

### Phase 4 first slice

- `replay.py` / `cli.py`: opt-in `replay --allow-cache`, matching API configuration preflight,
  cache readiness checks, authenticated namespace invalidation after warm-up, real cache-hit
  counts, embedding token totals, and saved judge context verified against API prompt hashes.
  Baseline-only behavior remains the default. Prefix reset is manual and disclosed.
- `evaluation.py`: `judge-replay` judges every saved semantic hit with policy/context and expected
  requirements; uncertainty, provider errors and unverifiable context fail closed. Exports JSONL,
  descriptive Wilson intervals, up to 50 manual-review cases, summary JSON and Markdown report.
- `compare`: requires successful disabled/exact/semantic runs with identical traffic, generation
  settings, serving metadata, warm-up counts and actual prompt sequences; exports CSV deltas and
  a report that leaves prefix warm-state equivalence unverified.
- `README.md`: smoke replay, judging and manual three-mode comparison commands with expected output.
  No live providers contacted; `.env` unchanged. Paid endpoint cost accounting remains unavailable.

### Phase 5 dashboard and handoff

- `src/cachewise/dashboard.py`: read-only Streamlit dashboard with process-scoped hit rates,
  avoided calls, failures, latency, observed token usage, active versions and cache namespaces.
  Per-response prefix token evidence is separate from application hits.
- Explicit selectors load existing replay manifests, validated comparison CSVs (with charts and
  download), and judge summaries including split/hash, false-hit counts, Wilson interval and
  manual-review status. Offline API and incomplete/invalid JSON leave saved evidence usable.
- Costs remain explicitly unavailable where upstream accounting is absent; no inferred savings,
  automatic cross-run equivalence, or population quality guarantee is presented.
- Streamlit 1.64.0 is locked in `uv.lock`. `compose.yaml` adds a loopback dashboard at port 8501,
  health check and read-only artifact mount without provider/admin credentials. `.env.example`
  documents dashboard settings; README covers native/Compose startup and a six-step walkthrough.
- `tests/test_dashboard.py`: four Streamlit AppTest checks cover empty metrics, distinct prefix
  evidence, offline access with judge uncertainty, comparison charts/downloads and corrupt files.

## Decisions and invariants

- Phase 2 used exact caching only and broader rule-based routing; phase 3 adds opt-in semantic
  reuse while preserving that router. Rules evaluate the whole
  utterance; unfamiliar vocabulary bypasses. English questions may request another response
  language. General personalized-product policy questions can qualify; actual personal requests
  always bypass and use current ownership-filtered fixtures.
- A future Jev router should preserve the existing eligibility outcomes, accept only a
  high-confidence `eligible` choice, and fail closed on low confidence or provider errors. Its
  threshold must be tuned and held-out/adversarial false-cache behavior evaluated before rollout.
- Normalize FAQ question whitespace and Unicode NFC only; preserve case, punctuation, numbers,
  and negation. Hash the actual normalized prompt, tier/region/language, generation endpoint,
  model/revision/settings, tool definitions (currently empty), fixture versions and Redis namespace.
- Set `CACHEWISE_GENERATION_MODEL_REVISION` when weights or serving chat template change behind
  an unchanged model name. Prompt-content changes automatically change identity.
- Redis cache counters are separate from fixture versions. Invalidation does not edit policy
  facts. Namespace state is atomically updated; writes retain the namespace captured before
  generation. Already in-flight requests may finish with their captured snapshot.
- Only nonempty generations ending with `stop` are stored. Hits have fresh request IDs, zero
  generation token usage and unavailable prefix evidence. Actual provider usage remains nullable;
  hit responses never inflate generation or prefix-cache counters.
- Cache failures preserve generation. Failed or truncated generations are not stored. Redis
  outages do not make an otherwise working generation path unready. Concurrent misses may generate
  more than once; first-writer TTL is preserved. Stampede suppression is deferred.
- No self-hosted dollar savings are invented. Prefix-cache evidence remains distinct from Redis
  hits. No live vLLM requests were made during phase 2 implementation, per user instruction.

- Semantic context hashes every actual prompt message except the question, plus the exact-cache
  identity constraints and conservative question guards. Changing actual context/policy content
  changes the semantic context even without a version edit. Embedding namespace includes endpoint,
  model, revision, dimension and vector format; update revision after hidden model changes.
- Semantic hits never populate either cache or extend TTL. Exact hits never call embeddings.
  Embedding/index outages preserve generation and exact-cache behavior. Only generated successful
  answers populate vector entries. Semantic entries retain the pre-generation namespace.
- No semantic threshold is approved yet. The configured 0.80–0.99 range supports the next phase's
  sweep; conservative guards reduce coverage but cannot establish correctness. Old embedding
  indexes remain until explicit cleanup, while their entries expire normally.

## Verification

- `CACHEWISE_TEST_REDIS_URL=redis://127.0.0.1:16379/0 .venv/bin/pytest -q`:
  **147 passed**, two existing FastAPI/Starlette deprecation warnings. Used a dedicated temporary
  Redis 8.2 container, removed after tests; all generation and embeddings were test-only.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`: passed.
- Real Redis tests verified filtered vector retrieval, cosine threshold rejection, first-writer
  TTL, logical/physical expiry, corrupt payload fallback, API miss → exact hit → semantic hit,
  language isolation, and policy/catalogue invalidation. The initial integration run exposed a
  RESP3 parser mismatch; pinning RESP2 fixed it and the full suite passed.
- Sandbox thread-based tests stalled and were stopped; full suite passed outside the sandbox.
- Prior phase lockfile and Compose checks passed; dependencies and Compose unchanged in phase 3.
  Full application container build and live vLLM/embedding calls were not run.

### Phase 4 verification

- Full regression suite: **144 passed, 7 Redis integration tests skipped**, two existing warnings.
  Subsequently added prompt-sequence comparison checks and a regression test; focused evaluation
  suite: **5 passed**. Ruff lint and formatting passed. Tests used mock providers only.
- Automated threshold sweep/freezing, manual-review ingestion, dedicated prefix experiment,
  configured price accounting, and final combined ship-gate reporting are not yet implemented.
  Human review and all live quality/performance evidence remain pending.

### Phase 5 verification

- `.venv/bin/pytest -q`: **149 passed, 7 Redis integration tests skipped**, two existing warnings;
  full suite run outside sandbox after restricted API thread tests stalled and were stopped.
- `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/cachewise-uv uv lock --check --offline` and
  `docker compose config --quiet`: passed.
- `docker compose build dashboard`: passed with frozen production dependencies.
- Temporary network-isolated dashboard container: Streamlit health endpoint returned HTTP 200;
  container removed after check. No live inference/embedding/judge endpoints contacted.
- Existing uncommitted phase 3/4 work preserved; user `.env` unchanged.

## Current status and next steps

1. Phase 3 implementation is complete. Configure a live embedding provider/model, its dimension
   and an explicit provisional threshold to exercise semantic mode using the README walkthrough.
   Default cache mode and the user's `.env` were not changed.
2. Phase 4 remaining: integrate automated judge-based threshold sweep (tuning split only), freeze threshold, evaluate
   held-out traffic, manual review, confidence intervals, comparative replay and report export.
   No below-1% false-hit claim is established by deterministic automated tests.
3. Foundation live baseline/prefix demonstration and phase 2 user-run live smoke remain unverified.
   `scripts/smoke_exact_cache.py` remains available for exact mode. Dashboard/handoff is implemented; it exposes existing evidence and outstanding phase 4 gates.

## Update rule

Update after material implementation, design decisions, verification or handoff. Distinguish
implemented, automated evidence and live evidence. Never record credentials or private data.
