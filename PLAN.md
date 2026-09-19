# Cachewise — Implementation Plan

## Summary

Build a weekend-scale Python application that demonstrates safe LLM response caching for an ecommerce support assistant. Use FastAPI, Redis with vector search, Streamlit, and a replay harness over 1,000 synthetic, production-shaped requests.

Use a live self-hosted model served by vLLM for answer generation, with automatic prefix caching enabled. Keep model integrations behind provider-neutral interfaces. The generation model and serving hardware will be selected in a later discussion; do not assume a model identifier or memory budget yet. Configure embeddings and judging separately, with credentials where required. Mock responses are limited to automated tests.

Deliver a working demo and an evaluation report showing hit rates, net inference savings, latency changes, semantic accuracy, and cache invalidation.

## 1. Application and provider interfaces

- Run the API, Redis, and dashboard locally through Docker Compose. Document vLLM startup on the selected serving hardware, with an optional Compose service when compatible; allow the inference server to run on a separate machine. Provide environment-variable configuration and documented startup commands.
- Define separate interfaces for answer generation, embeddings, and judging. Implement an OpenAI-compatible HTTP adapter for each, with configurable base URLs, credentials, and model identifiers. Providers with different protocols require additional adapters.
- Point the answer-generation adapter at the vLLM OpenAI-compatible `/v1` endpoint with a configurable served model name and authentication settings. Embedding and judge model/provider choices remain open.
- Integrate vLLM prefix-cache usage reporting through response usage fields where available and engine metrics. Distinguish per-request evidence from aggregate server metrics; unsupported or unavailable metrics must display as unavailable.
- Configure prices explicitly for any paid embedding or judge endpoints. Report self-hosted generation tokens, avoided calls, and latency without assigning hosted API token prices. Dollar savings for self-hosted generation require an explicit, documented serving-cost model; otherwise display them as unavailable. Record all cost assumptions with each evaluation.
- Expose `POST /chat`, `POST /admin/invalidate`, `GET /metrics`, and `GET /health`. Protect invalidation with a local admin token.
- Return the answer, request identifier, and cache outcome from `/chat`. Use predefined demo customer profiles to resolve tier, region, and customer scope on the server.

## 2. Cache behavior and safety

### Request flow

1. Resolve the customer profile and applicable policy versions.
2. Decide whether the request is eligible for response caching.
3. Look up the exact cache.
4. On an exact miss, embed the question and search eligible semantic entries.
5. Reuse a response only when its similarity meets the configured threshold.
6. Otherwise call the vLLM generation endpoint, where the engine can reuse cached prompt prefixes, then store eligible responses and record usage and latency.

### Eligibility and identity

- Cache shared FAQ responses covering returns, shipping, and payment methods. Bypass response caching for personalized requests, live stock questions, and conversations containing previous turns.
- Use a conservative eligibility router; uncertain requests bypass caching. Test it against personalized requests phrased like FAQs.
- Build exact keys from the normalized question plus model identity, generation settings, tool definitions, customer tier, region, language, system-prompt hash, and policy/catalogue versions.
- Apply the same context constraints as mandatory filters before semantic retrieval. Include embedding model and dimension in the semantic index namespace.
- Normalize whitespace and Unicode conservatively; preserve case, punctuation, numbers, and negation.
- Use cosine similarity and one configurable semantic threshold for the initial version. Cache generated answers only; do not create new entries from semantic hits.
- Keep demo order data in a small fixture-backed lookup service. Personalized answers always obtain current fixture data and never enter the shared response cache.

### Expiration and invalidation

- Default shared FAQ TTL to 24 hours, without extending expiry on hits. Reject expired semantic entries even if Redis has not removed them from the index yet.
- Increment namespace versions on policy changes and catalogue reindex events. Derive prompt identity from the actual prompt contents so a prompt edit automatically causes misses.
- Associate a request’s writes with the versions captured at its start, preventing an in-flight response from entering a newer namespace.
- Treat cache failures as misses and continue to generation. Embedding failures bypass semantic lookup; generation failures return an error and are never cached.

### vLLM automatic prefix caching

- Enable automatic prefix caching explicitly with `vllm serve <model-id-or-local-path> --served-model-name cachewise-model --enable-prefix-caching`. Final model, precision, context length, and memory settings depend on the later model/hardware selection.
- Place static instructions, stable policy text, and tool definitions before variable customer and question content. Keep prompt serialization and the chat template consistent so requests share an identical token prefix.
- Let vLLM manage the KV prefix cache. Redis exact/semantic caches reuse complete answers; vLLM prefix caching reuses prompt computation on requests that reach generation, including response-cache bypasses. It reduces prefill work but still generates a fresh answer.
- Record engine-reported prefix reuse and generation usage where available. Prefix cache residency is engine-managed and separate from Redis TTLs and version namespaces; changed policy or prompt contents must be reflected in the actual generated prompt.
- Demonstrate reuse with repeated long prefixes and different questions, using engine evidence and a separate enabled-versus-disabled latency experiment. Do not infer cache hits solely from faster responses or claim a benefit without measured evidence.

## 3. Replay, evaluation, and dashboard

### Dataset and replay

- Create a seeded dataset of 1,000 requests with repeated FAQs, paraphrases, difficult near-matches, different tiers and regions, and personalized questions. Treat the brief’s 60% repetition as a dataset assumption, not an expected hit rate.
- Include expected intent, context, and answer requirements. Use fictional customer data only.
- Split traffic into a 300-request tuning set and a 700-request held-out evaluation set. Keep paraphrase families separate across the split while preserving repetition within each set.
- Tune on the tuning set, freeze the threshold, then evaluate once on held-out traffic. Also report the requested full 1,000-query replay, clearly distinguishing it from held-out evidence.
- Compare no application cache, exact-only caching, and exact-plus-semantic caching. Reset cache state between configurations and replay the same ordered traffic with concurrency fixed at one.
- Measure real end-to-end latency, embedding usage, generation usage, and vLLM prefix usage. Keep prefix caching enabled across the three application-cache comparisons. Start each configuration with a reset vLLM prefix cache or a fresh server and the same warm-up procedure; disclose warm-state effects.
- Run a separate prefix caching experiment with application response caching disabled, comparing vLLM prefix caching enabled and disabled on identical ordered requests with shared prefixes. Record time to first token when measurable, end-to-end latency, and engine prefix-cache evidence. Keep generation settings and concurrency fixed.

### Semantic quality

- Sweep cosine thresholds from 0.80 to 0.99 in increments of 0.01. Plot semantic precision and coverage against threshold.
- Have the live judge assess semantic reuse against the new request, applicable policy, and expected answer requirements. Similarity to a fresh generated answer alone is insufficient.
- Judge all semantic hits in the final replay, given the small dataset size. Count unparseable or uncertain judgments as failures.
- Manually review 50 labeled cases, including difficult matches and judge failures, to check judge reliability.
- Select the threshold with the highest tuning hit rate that stays below 1% observed false hits. If none qualifies, disable semantic reuse and report the unmet gate.
- Report the false-hit numerator, denominator, and a 95% confidence interval. Zero observed failures with a small sample does not establish a population false-hit rate below 1%.

### Reporting

- Provide a Streamlit dashboard showing exact and semantic hit rates, avoided generation calls, baseline versus cached usage and available cost estimates, p50/p95 latency, semantic false-hit rate, and active versions.
- Include paid embedding costs in net monetary savings when a generation serving-cost model is available. Otherwise report usage reductions and paid endpoint costs separately, with generation dollar savings unavailable. Report judge costs separately as evaluation overhead. Disclose whether serving infrastructure costs are excluded or included in an explicit cost model.
- Present vLLM prefix reuse separately from application response-cache hits and report the dedicated prefix experiment separately from the application-cache comparisons.
- Export per-request JSONL results, aggregate CSV metrics, the precision curve, and a Markdown report. Record dataset seed, configuration, models and revisions, chat template, vLLM version, serving hardware, precision, context length, prefix settings, prices/cost assumptions, and policy/prompt versions for reproducibility.

## 4. Implementation order and acceptance tests

1. **Foundation:** configuration, provider adapters, vLLM serving setup with prefix caching, fixture support assistant, and live baseline replay after model/hardware selection.
2. **Exact caching:** context-aware keys, TTLs, version invalidation, and cache-failure fallback.
3. **Semantic caching:** embeddings, filtered vector search, eligibility routing, and threshold configuration.
4. **Evaluation:** judge integration, threshold sweep, held-out replay, separate prefix caching experiment, and report export.
5. **Dashboard and handoff:** readable metrics, Docker startup, environment template, and walkthrough.

Required verification:

- Equivalent exact requests hit; changes to any answer-affecting context miss.
- Approved paraphrases can hit; negations, changed destinations, and changed conditions are included as adversarial cases.
- Two customers asking about their orders receive their own fixture data, with no shared response-cache hit.
- TTL expiry, policy updates, catalogue updates, and actual prompt edits prevent reuse of old answers in both caches.
- Redis or embedding outages preserve the generation path; failed generations never populate caches.
- Replay reports measured hit rate, generation/embedding usage delta, available cost delta, and p50/p95 latency delta for 1,000 queries. Unsupported self-hosted dollar estimates are marked unavailable.
- Final semantic false-hit rate is below 1% on evaluated hits; sample uncertainty remains visible.
- vLLM prefix caching is demonstrated with engine-reported evidence and an enabled-versus-disabled latency comparison, or marked as an outstanding gate.
- The final report marks each ship gate as passed, failed, or insufficient evidence. No savings or latency target is invented beyond the supplied requirements.

## Assumptions and boundaries

- The initial release is a local demonstration with fictional data and fixture authentication.
- Single-turn shared FAQs are the only responses eligible for semantic reuse.
- vLLM is the selected generation inference engine. Generation model and serving hardware selection are deferred to a later discussion and are prerequisites for live implementation/evaluation. Verify model support and capacity for weights plus KV cache before finalizing serving settings.
- Embedding and judge model/provider selection, compatible endpoints, credentials where required, and any pricing/cost assumptions are deployment configuration prerequisites.
- Real ecommerce integration, production authentication, distributed invalidation infrastructure, and load testing are deferred.
