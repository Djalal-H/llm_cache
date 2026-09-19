# Cachewise

An ecommerce support assistant with fictional customer/order fixtures, optional safe exact
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

## Docker API

```bash
docker compose up --build
```

The API-only service uses Linux host networking for dependency downloads during its build and
to reach native vLLM at `127.0.0.1:8001`, and binds
the API at `127.0.0.1:8000`. Stop a native API process first to avoid a port conflict. vLLM runs
outside this container. Host networking must be supported/enabled to use this configuration.
Redis runs in its own service with a loopback-only port and a persistent volume.
Semantic caching, judge evaluation, and Streamlit arrive in later phases.

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
