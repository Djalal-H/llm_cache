# Cachewise

An ecommerce support assistant with fictional customer/order fixtures, optional exact and semantic
response caching in Redis, independent OpenAI-compatible provider adapters, and a sequential
baseline replay. Caching defaults to disabled; enable `CACHEWISE_CACHE_MODE=exact` for shared FAQs.

## Local startup

Requirements: Linux, `uv`, an NVIDIA driver exposing CUDA, and network access to PyPI and Hugging
Face. The selected target is the RTX 3060 Laptop GPU with 6 GB VRAM and
`Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`. Application and serving environments are separate.

Install the API dependencies and create local configuration:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
uv python install 3.12
uv sync --frozen
cp .env.example .env
```

Install the pinned serving dependencies, then start vLLM in its own terminal:

```bash
bash scripts/setup_vllm.sh
mkdir -p artifacts/diagnostics
.venv-vllm/bin/python scripts/serve_vllm.py 2>&1 | tee artifacts/diagnostics/vllm-startup.log
```

The launcher records the resolved model/tokenizer revision, template hash, hardware, and startup
settings in `artifacts/serving.json`. It uses vLLM 0.29.0, GPTQ Int4 weights, FP16 compute, a
2,048-token context, 85% GPU memory utilization, eager execution, one concurrent sequence, and
automatic prefix caching. These are initial settings: successful startup must confirm space for
weights, runtime allocations, and the KV cache. A startup manifest alone does not prove readiness.
No alternate model or CPU fallback is selected on failure.

Start the API in a second terminal:

```bash
uv run --frozen uvicorn cachewise.api:app --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/metrics
```

If running from a restricted coding sandbox, GPU access, downloads, and local network operations
may require execution outside that sandbox. A failed sandbox `nvidia-smi` call alone does not
establish a driver failure.

## Support API

```bash
curl --fail http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"demo-alice","question":"Where is my order?","language":"en"}'
```

Demo profiles are `demo-alice` (standard, US), `demo-bob` (premium, US), `demo-claire` (standard, EU),
and `demo-dan` (premium, EU). Choosing a fixture identifier is the demo's authentication mechanism.
Tier and region are resolved on the server and cannot be supplied by the client.

`POST /chat` accepts `customer_id`, `question`, optional `language` (default `en`), and optional
`history`, a list of `{ "role": "user" | "assistant", "content": "..." }` objects. It returns
the answer, request identifier, cache outcome, usage, finish reason, model, prompt hashes and
captured policy/catalogue versions. The `X-Request-ID` header correlates unsuccessful requests too.
Unknown customers receive 404; invalid requests receive 422; generation failures receive 502 and
timeouts receive 504. Overlength model prompts are returned as upstream errors rather than silently
truncated. No live mock responses are available.

Personalized requests load current orders from fixtures, filtered by customer ownership before
generation. Shared FAQ prompts omit customer names, identifiers and order lookup data. Policies are resolved for the customer's tier and region, including premium shipping
overrides for each supported destination. Instructions and these stable, versioned policy facts
precede variable history, customer details,
order lookup and question content. `system_prompt_hash` identifies the actual static contents;
`serialized_prompt_hash` identifies the complete application message serialization, before the
model's chat template is applied. Dates and tracking references are fictional.

`GET /health` returns 503 when the configured generation model or fixtures are unavailable.
`GET /metrics` exposes process-local counts and a latency window covering the last 10,000 chat
requests. Known usage totals include observation counts; unavailable usage is never assumed zero.
Embedding and judge endpoints can be configured independently in `.env`; neither is called by
the foundation chat path. Their interfaces and adapters are ready for later phases.

## Exact caching

Set these values in `.env` (choose your own admin token):

```dotenv
CACHEWISE_CACHE_MODE=exact
CACHEWISE_REDIS_URL=redis://127.0.0.1:6379/0
CACHEWISE_CACHE_TTL_SECONDS=86400
CACHEWISE_CACHE_TIMEOUT_SECONDS=1
CACHEWISE_ADMIN_TOKEN=replace-with-your-local-admin-token
```

Start Redis, then start/restart the API so it reads the new configuration:

```bash
docker compose up -d redis
uv run --frozen uvicorn cachewise.api:app --host 127.0.0.1 --port 8000
```

Keep the native vLLM server running in its own terminal as described above. In a second terminal,
from the repository root, run the live smoke test yourself:

```bash
uv run --frozen python scripts/smoke_exact_cache.py
```

The script reads the admin token from `.env`. It invalidates both cache namespaces first so the
run is repeatable, sends the same FAQ twice, invalidates the policy namespace, and sends the FAQ
again. It prints answers, request IDs, outcomes, measured latency, usage, namespace versions,
and metric deltas. With no other traffic, expect `miss`, `exact_hit`, `miss`, two generation calls,
one avoided generation call, and zero cache errors. It exits nonzero when checks fail. This is a
live generation test, provided for manual execution; it is not run by pytest. Invalidation affects
all FAQ entries in the selected namespace on this demo instance.

`POST /admin/invalidate` accepts `{"scope":"policy"}`, `{"scope":"catalogue"}`, or
`{"scope":"all"}` with `Authorization: Bearer <admin-token>`. A blank configured token disables
the endpoint (404); incorrect credentials return 401 and unavailable Redis returns 503. It changes
cache versions only: edit fixture policies/catalogue versions separately when facts change. Old
entries expire naturally; there is no database flush. In-flight generation keeps its original
namespace and cannot populate the new one. Requests already in flight may finish with their
captured policy snapshot.

The rule-based router recognizes English FAQ wording about returns, shipping, and payments using
intent rules and reviewed policy vocabulary. Requested response language is independent and part
of the key. Unrecognized wording, mixed requests, previous turns, personal orders/charges/refunds,
and live stock bypass caching. General questions about personalized products may qualify.
Eligibility is conservative and does not establish the later semantic-quality gate.

Exact identity includes the normalized question, actual prepared prompt, server-resolved tier and
region, requested language, model/endpoint/revision, generation settings, empty tool definitions,
and fixture/cache versions. Normalization uses Unicode NFC and collapsed whitespace, preserving
case, punctuation, numbers, and negation. Set `CACHEWISE_GENERATION_MODEL_REVISION` whenever weights
or the serving chat template change behind the same served model name. API keys and admin tokens
are never included in exported configuration.

Only nonempty generations ending with `finish_reason: "stop"` are stored. Hits do not extend the
24-hour default TTL. `miss` includes eligible cache lookups that failed; `bypass` means disabled
caching or an ineligible request. Redis failure preserves generation; write failure preserves the
answer. A hit reports zero generation tokens and unavailable prefix evidence, with a fresh request
ID. Actual generation retains nullable provider usage. Metrics count provider usage only for real
generation calls and expose hits, misses, bypass reasons, cache errors, and active namespaces.
Redis health is reported separately and does not make a working generation path unready.

Concurrent misses may generate more than once; stampede suppression is deferred. Redis persists
namespace metadata alongside entries. Recovery from missing/corrupt namespace state creates a
fresh identity so surviving old entries cannot be reused. Shared FAQ prompt construction also
applies with caching disabled, preserving comparable generation inputs for future evaluation.

## Semantic caching (phase 3)

Set `CACHEWISE_CACHE_MODE=semantic` to enable exact-first, then semantic lookup. Configure
`CACHEWISE_EMBEDDING_BASE_URL`, `CACHEWISE_EMBEDDING_MODEL`, credentials when required,
`CACHEWISE_EMBEDDING_DIMENSION` (the endpoint's actual output size), and
`CACHEWISE_SEMANTIC_THRESHOLD` explicitly. Thresholds must be between 0.80 and 0.99;
there is no default or quality-approved threshold yet. The existing OpenAI-compatible embedding
adapter is used; model and provider selection remain deployment choices. Leave caching disabled
or exact-only until you are ready to test semantic reuse. No judge is called in the chat path.

Redis 8.2 in Compose supplies the vector search engine. The implementation uses a FLOAT32
FLAT cosine index with mandatory context and expiration filters before nearest-neighbor selection,
following the [Redis vector search documentation](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/).
Each embedding endpoint/model/revision/dimension combination has its own index. A context digest
covers every prompt message except the question, generation identity/settings, tier, region,
language, fixture versions, and the captured invalidation namespace. Conservative question filters
also separate negations, numbers, explicit destinations, and common return/payment conditions.
These filters deliberately reduce coverage and do not establish semantic correctness.

On an exact miss, the API embeds the normalized question and retrieves the nearest eligible
entry. Reuse requires cosine similarity at or above the configured threshold. Only successful,
nonempty, completed generations populate either cache. Semantic hits never create exact or vector
entries and never refresh TTLs. Both caches share `/admin/invalidate`; in-flight generations retain
the namespace captured before generation. Cache/index failures and embedding failures preserve
generation; embedding failures still allow exact-cache writes. Redis without vector support reports
`semantic_cache: unavailable` in `/health` while generation and exact caching remain usable.

`/chat` now reports `semantic_hit`, optional `semantic_similarity` and `semantic_source_identity`,
and separate `embedding_usage`. Hits report zero generation tokens and unavailable prefix evidence.
`/metrics` separates exact/semantic hits, embedding calls/errors and known embedding token usage;
avoided generation calls include both hit types. Unknown provider usage remains unavailable.
`CACHEWISE_EMBEDDING_MODEL_REVISION` must change when embedding behavior changes behind an
unchanged model name. Old indexes remain until explicitly cleaned up; their entries expire normally.

For a manual live check, start Redis and the API with those settings, then send an eligible FAQ
(e.g. “How much does shipping cost?”), repeat it, and send a paraphrase (“What does shipping cost?”).
Expect a miss, an exact hit, then a semantic hit only if the live embedding similarity passes your
threshold. Check `/metrics`, invalidate, and retry to confirm a new generation. Actual models may
produce a semantic miss; never lower the threshold solely to force a demo hit.

The next phase sweeps thresholds on tuning traffic and evaluates held-out semantic hits with a
live judge. The below-1% false-hit gate, manual review, and confidence interval remain unverified.
Automated tests use deterministic providers and real Redis vectors, without calling live LLMs.

## Baseline replay

Set `CACHEWISE_CACHE_MODE=disabled` for both the API and replay runner, restarting the API if
necessary. Replay preflight rejects enabled caching before any warm-up request.
Generate the deterministic dataset and run a small live replay first:

```bash
uv run --frozen cachewise dataset --seed 42
uv run --frozen cachewise replay --limit 10 --output artifacts/baseline-smoke \
  --serving-metadata artifacts/serving.json
```

Inspect answers for FAQ correctness and customer-owned order grounding, including both Alice's
and Bob's order questions. After the smoke check, stop and restart vLLM to obtain fresh prefix
state, then run the full baseline:

```bash
uv run --frozen cachewise replay --output artifacts/baseline-full \
  --serving-metadata artifacts/serving.json
```

Each replay is sequential, with two unmeasured warm-up calls by default. A fresh output directory
is required to prevent evidence overwrite. `--split tuning|evaluation|all`, `--limit`,
`--warmup-count`, and `--api-url` are available. The 1,000-request dataset has exactly 60% exact
repetitions, 300 tuning rows and 700 evaluation rows; paraphrase families do not cross the split.
Repetition is a dataset assumption, not a promised cache hit rate. Baseline generation does not
tune a threshold or establish semantic quality.

Exports contain `requests.jsonl`, `aggregate.csv`, `warmup.jsonl`, `manifest.json`, and raw engine
metric snapshots when reachable. Failed requests remain in the JSONL, and failed preflight checks
preserve a manifest. Latency includes API/provider transport and generation; p50/p95 for all
requests and successful requests are reported separately. Truncated answers are counted.
The CLI exits nonzero if any measured request fails.

Per-request prefix evidence is taken only from `usage.prompt_tokens_details.cached_tokens`.
Engine prefix metrics are preserved separately as aggregate snapshots; they cannot identify an
individual request hit. Unsupported evidence remains unavailable. Prefix caching remains enabled
for this baseline; an enabled-versus-disabled comparison belongs to the later evaluation phase.
Self-hosted generation dollars remain unavailable because no serving-cost model has been supplied.
The provided serving manifest describes startup settings; retain the startup log as readiness and
capacity evidence, and do not pair a stale manifest with another server.

## Docker application

```bash
docker compose up --build
```

The API and dashboard services use Linux host networking for dependency downloads during its build and
to reach native vLLM at `127.0.0.1:8001`, and binds
the API at `127.0.0.1:8000`. Stop a native API process first to avoid a port conflict. vLLM runs
outside this container. Host networking must be supported/enabled to use this configuration.
Redis runs in its own service with a loopback-only port and a persistent volume.
Open the Streamlit dashboard at http://127.0.0.1:8501. It reads the API metrics and mounts
`./artifacts` read-only for saved evaluation results. The dashboard does not need provider or admin
credentials. Stop a native dashboard first if port 8501 is already occupied.

## Verification

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Automated tests use test-only providers and HTTP transports. They check ownership boundaries,
current fixture reads, profile resolution, prompt stability/changes, malformed/missing usage,
provider failures, API validation, deterministic dataset partitioning, replay failures and exports.
Foundation acceptance additionally requires successful live startup, representative live answers,
and the full 1,000-request baseline; automated tests alone do not satisfy that gate.

Real Redis integration tests use isolated keys and test-only generation. To run them separately:

```bash
docker run --rm -d --name cachewise-exact-tests -p 127.0.0.1:16379:6379 \
  redis:8.2-alpine redis-server --save '' --appendonly no
CACHEWISE_TEST_REDIS_URL=redis://127.0.0.1:16379/0 uv run --frozen pytest -q
docker stop cachewise-exact-tests
```

Without `CACHEWISE_TEST_REDIS_URL`, the Redis integration tests skip explicitly. The suite never
runs the manual live smoke script or contacts vLLM. Redis adapter lifecycle and atomic namespace
operations follow the [redis-py asyncio documentation](https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html)
and [Redis EVAL documentation](https://redis.io/docs/latest/commands/eval/).

## Phase 4: replay and judge evaluation

The first evaluation slice adds cached replay, saved-hit judging, a manual-review queue, and
three-mode comparison. Automatic threshold sweeping/freezing, a dedicated prefix experiment,
price accounting, and the final combined ship-gate report are still pending. A smoke run does
not approve a semantic threshold. Do not use held-out traffic to tune configuration.

Use the existing generation/embedding setup. For semantic mode, `.env` must contain the embedding
endpoint, model, dimension, a provisional threshold, and a nonempty `CACHEWISE_ADMIN_TOKEN`.
For judging, set `CACHEWISE_JUDGE_BASE_URL`, `CACHEWISE_JUDGE_MODEL`, and
`CACHEWISE_JUDGE_API_KEY` if required. Judge calls use the configured live endpoint and can incur
charges. This version exports token usage but does not compute endpoint costs.

Create the dataset and start Redis:

```bash
.venv/bin/cachewise dataset --output artifacts/evaluation-dataset.jsonl
docker compose up -d redis
```

Start the API in a separate terminal (stop any existing API on port 8000 first):

```bash
CACHEWISE_CACHE_MODE=semantic .venv/bin/uvicorn cachewise.api:app --host 127.0.0.1 --port 8000
```

Run a small tuning-only smoke replay from the repository root:

```bash
CACHEWISE_CACHE_MODE=semantic .venv/bin/cachewise replay \
  --allow-cache --dataset artifacts/evaluation-dataset.jsonl \
  --split tuning --limit 25 --warmup-count 2 --output artifacts/semantic-smoke

.venv/bin/cachewise judge-replay \
  --run artifacts/semantic-smoke --output artifacts/semantic-smoke-judged
```

Expect `Replayed 25/25 requests` followed by aggregate JSON with exact/semantic hit counts,
avoided generation calls, embedding tokens, generation usage and measured latency. The first
eligible request normally misses; repeats can hit. Zero semantic hits is a valid smoke outcome,
particularly with a small sample or conservative routing. The judge prints `Judged N/M semantic
hits`, then the false-hit numerator/denominator, observed rate and 95% Wilson interval. With no
semantic hits, the gate is `insufficient evidence`. Missing policy context, uncertain judgments,
and judge errors count as failed judgments. A passed observed gate is not a population guarantee.

Replay writes `requests.jsonl`, `aggregate.csv`, `manifest.json`, `warmup.jsonl` and available engine
snapshots. Judging writes `judgments.jsonl`, `summary.json`, `manual-review.csv` and `report.md`.
Review the CSV manually; it contains up to 50 hits, prioritizing judge failures and low similarity.
Fewer than 50 cases leaves the manual-review target unmet. Every output directory must be new.

`--allow-cache` uses the cache mode already configured in the API and runner. The two configurations
must match. It invalidates both application-cache namespaces **after warm-up**, making measurement
start with an empty response-cache namespace without deleting Redis data. Use a dedicated demo
instance with no other traffic. Prefix cache is not reset automatically: restart vLLM before each
comparison run, keep prefix caching enabled, and use identical warm-up counts and serving settings.

For a tuning comparison, repeat the API startup and replay for each mode (`disabled`, `exact`,
`semantic`), using the matching environment variable in both terminals:

```bash
CACHEWISE_CACHE_MODE=disabled .venv/bin/cachewise replay --allow-cache \
  --dataset artifacts/evaluation-dataset.jsonl --split tuning --output artifacts/tuning-disabled
CACHEWISE_CACHE_MODE=exact .venv/bin/cachewise replay --allow-cache \
  --dataset artifacts/evaluation-dataset.jsonl --split tuning --output artifacts/tuning-exact
CACHEWISE_CACHE_MODE=semantic .venv/bin/cachewise replay --allow-cache \
  --dataset artifacts/evaluation-dataset.jsonl --split tuning --output artifacts/tuning-semantic

.venv/bin/cachewise compare \
  --runs artifacts/tuning-disabled artifacts/tuning-exact artifacts/tuning-semantic \
  --output artifacts/tuning-comparison
.venv/bin/cachewise judge-replay \
  --run artifacts/tuning-semantic --output artifacts/tuning-semantic-judged
```

Do not run those three replay commands against a single unchanged API configuration. Pass the same
`--serving-metadata` file to each replay if available. Comparison rejects differing traffic,
generation settings, serving metadata, warm-up counts or failed runs. Negative deltas in
`comparison.csv` mean fewer tokens/calls or lower latency than the disabled-cache baseline.
Prefix warm-state equivalence remains explicitly unverified; no latency improvement is promised.
The baseline-only behavior of `replay` without `--allow-cache` is preserved.


## Dashboard and demo walkthrough

For a native dashboard, run from the repository root:

```bash
uv sync --frozen
uv run --frozen streamlit run src/cachewise/dashboard.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open http://127.0.0.1:8501. Use **Refresh** to reread process metrics and local artifacts.
`CACHEWISE_DASHBOARD_API_URL` and `CACHEWISE_DASHBOARD_ARTIFACTS_DIR` can be exported to
select a different API or evidence directory; native Streamlit does not load these from `.env`.
Compose reads the API URL from `.env` and always mounts `./artifacts` at `/app/artifacts`.
The dashboard remains usable for saved evidence when the API is offline.

1. Start vLLM using the local startup instructions, configure exact or semantic mode in `.env`,
   then run `docker compose up --build -d`. Check `docker compose ps` and the API `/health`.
2. Open the dashboard. Send the same shared FAQ twice using the Support API example with
   `"question":"What is the return policy?"`. In exact or semantic mode, a successful cacheable
   generation followed by a repeat should increment exact hits and avoided generation calls.
   Refresh to inspect counters, token observation coverage, p50/p95 latency and active versions.
3. Ask `"Where is my order?"` as both `demo-alice` and `demo-bob`. Inspect each response's fixture
   data and bypass outcome. Use the earlier invalidation example to increment a namespace;
   refresh and repeat the FAQ to observe a miss in the new namespace.
4. Follow the evaluation commands above to generate replay manifests, a validated three-mode
   `comparison.csv`, and judge `summary.json`. Select each explicitly in the dashboard. Replay
   tables show usage and failures; comparison charts show hit rates, tokens and latency. Judge
   evidence includes its dataset hash and split, false-hit numerator/denominator and 95% interval.
   Ensure selected judge and replay datasets match before interpreting them together.
5. Expand reproducibility evidence to inspect saved engine snapshots and configuration. Download
   the selected manifest or comparison CSV. Request JSONL, judge labels and manual-review CSV
   remain in their original artifact directories for auditing.
6. Stop application services with `docker compose down`; the Redis volume and local artifacts
   persist. Stop the separately running vLLM process when finished.

Live counters cover one API process since startup, with a 10,000-request latency window including
failures. Hit-rate denominators include all requests. Known token totals can be partial; observation
counts are shown, and missing usage is never interpreted as zero. Replay latencies are end-to-end
measurements and have a different scope from live API timing. Prefix reuse is displayed separately;
engine snapshots can include unrelated traffic and do not establish a latency benefit on their own.

Phase 5 dashboard/handoff is implemented. Phase 4 still needs automated threshold selection and
freezing, a held-out quality gate, human review ingestion, configured paid cost accounting, and a
dedicated prefix enabled/disabled experiment. No final ship approval or savings claim follows from
the dashboard. Generation/net dollar savings remain unavailable without a serving-cost model;
judge expenses are evaluation overhead. The UI surfaces these missing gates and uncertainty.
