# POC Results

*This document captures the latency numbers and early findings from Phase 0 (Day 1) POC A and POC B.*

## POC A — Qdrant Hybrid Round-Trip
**Goal:** Verify Qdrant boot, collection creation with dense and sparse vectors, ingestion of toy chunks, and one query combining `prefetch(dense) + prefetch(sparse)` with RRF fusion.

- **Status:** Infra verified; search/latency not yet measured (Docker booted successfully, collection `poc_collection` initialized)
- **End-to-End Latency:** *Search step pending implementation in Phase 1*
- **Observations:** Qdrant container boots clean on the `docker compose up -d` command. The Python script connects instantly and successfully configures the `dense` and `sparse` configurations using the SDK.

## POC B — LLM Provider Smoke Test
**Goal:** Send 5 sample prompts (one decomposition-style, one synthesis-style, one controller-classification-style) to the default provider and verify JSON parseability and latency.

- **Provider:** Gemini
- **Model:** `gemini-3.8-flash`
- **Total Latency:** ~3400ms – 3600ms for short responses (up to 8100ms for large generated output).
- **JSON Parsed Correctly:** [x] Yes (Prompt 5 successfully generated valid JSON).
- **Observations:** 
  - `gemini-1.5-flash` is deprecated on the v1beta API endpoint used by the SDK, so we upgraded to `gemini-3.8-flash`.
  - The API successfully parsed structured requests (e.g., JSON intent, decomposition lists).
  - Rapidly firing 5 concurrent/sequential prompts on the free-tier SDK resulted in two `503 UNAVAILABLE` (High Demand) errors. **Actionable finding:** We will need to implement basic retry/exponential backoff logic or use an enterprise key when running the heavy eval tests (Gate 6).
