# Implementation state

## Project snapshot

Cachewise is a Python 3.12 FastAPI foundation for a fictional ecommerce support assistant. The current implementation always calls a live OpenAI-compatible vLLM generation endpoint; application response caching is not implemented yet. The planned next phases are exact caching, semantic caching, evaluation, and a dashboard.

## Completed

- Added this dedicated continuity file and installed the `implementation-continuity` Codex skill. Future progress/resume requests should read and update this file before broad repository exploration.
- Reviewed the foundation architecture and the request path: `api.py` → `assistant.py` → `fixtures.py` → `providers.py` → vLLM.
- Reviewed the core contracts and configuration: `models.py`, `config.py`, and `metrics.py`.
- Reviewed fixture ownership and policy resolution in `fixtures.py`, prompt assembly and hashes in `assistant.py`, and provider adapters in `providers.py`.
- Reviewed FastAPI lifecycle/middleware behavior and the API-only Docker deployment boundary.
- Identified the local demo startup path: native vLLM on `127.0.0.1:8001`, API on `127.0.0.1:8000`; the single-request terminal client is `curl` because `cachewise` CLI currently only offers `dataset` and `replay`.

## Decisions and invariants

- The repository-local state file is `.codex/implementation-state.md`. It is a concise handoff record, not a substitute for source, tests, or user requirements.
- Customer tier, region, policies, and order lookup are server-resolved. Orders are filtered by customer ownership before prompt construction.
- Prompt identity is recorded with static `system_prompt_hash` and full application-message `serialized_prompt_hash`.
- Missing provider usage is represented as unavailable (`null`), never zero.
- The Docker image packages only the API; `scripts/serve_vllm.py` starts vLLM natively and requires compatible NVIDIA/CUDA hardware.

## Verification

- Source inspected; no code changes or test runs performed.
- Live service has not been started or checked in this session.

## Current status

- Current learning walkthrough has covered `models.py`, `fixtures.py`, `assistant.py`, `providers.py`, `api.py`, `metrics.py`, and `config.py`.
- Remaining useful walkthrough targets: `dataset.py`, `replay.py`, `cli.py`, `scripts/serve_vllm.py`, and tests.
- Next implementation work should begin from the requested phase in `PLAN.md`; response caching remains intentionally absent from the foundation.
