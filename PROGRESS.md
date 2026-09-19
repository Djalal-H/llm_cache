# Cachewise implementation progress

Last updated: 2026-09-19

## Project snapshot

**Phase 2 — exact response caching implemented and automatically verified.** Live vLLM verification
is intentionally reserved for the user. The application defaults to `CACHEWISE_CACHE_MODE=disabled`;
`exact` enables Redis caching for eligible shared FAQs. Semantic caching is the next implementation
phase in `PLAN.md`.

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

## Decisions and invariants

- User selected exact caching only and broader rule-based routing. Rules evaluate the whole
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

## Verification

- `CACHEWISE_TEST_REDIS_URL=redis://127.0.0.1:16379/0 .venv/bin/pytest -q`:
  **119 passed**, with two existing FastAPI/Starlette deprecation warnings. Used a dedicated
  temporary Redis 8.2 container and test-only generation, including the API cache sequence.
- Coverage includes FAQ/adversarial routing, shared-customer isolation, identity changes,
  current personalized fixture data, TTL/no-refresh, malformed entries, cache outages, generation
  failures/truncation, auth, atomic invalidation, in-flight generation and namespace recovery.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`: passed.
- `UV_CACHE_DIR="$PWD/.cache/uv" uv lock --check --offline`: passed.
- `docker compose config --quiet`: passed. Full application container build not rerun.
- Restricted sandbox TestClient/thread checks stalled; suite succeeded outside the sandbox.
  One approval-review attempt timed out; its permitted retry succeeded. No blocked action remains.

## Current status and next steps

1. User sets `CACHEWISE_CACHE_MODE=exact` and a nonempty `CACHEWISE_ADMIN_TOKEN` in `.env`, starts
   Redis and restarts the API with their live vLLM server running.
2. User runs `uv run --frozen python scripts/smoke_exact_cache.py` from the repository root.
   It invalidates the demo FAQ namespaces before testing; expect two generations, one exact hit,
   two misses and zero cache errors with no concurrent traffic. Record the user's live result here.
3. Foundation live baseline/prefix demonstration remain unverified in this session. Semantic
   caching, threshold evaluation, comparative replay, and dashboard/handoff remain future work.

## Update rule

Update after material implementation, design decisions, verification or handoff. Distinguish
implemented, automated evidence and live evidence. Never record credentials or private data.
