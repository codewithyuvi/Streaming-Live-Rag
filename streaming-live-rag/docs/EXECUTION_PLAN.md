# 9-Day Execution Plan

This is a summary of the Day-by-Day Phase Plan. Do not start the next phase's tasks until the current phase's gate is checked off.

## Phase 0 — Day 1: Technology Research, Architecture Lock & Foundation
**Goal:** Lock tech stack, stand up skeleton repo, Docker, shared schemas.
- Complete 5 ADRs.
- Execute POC A (Qdrant hybrid round-trip) & POC B (LLM provider smoke test).
- Scaffold repo, define Pydantic schemas.
- Start `docs/RISKS.md`.
**Gate 1 (informal):** `docker compose up` boots clean on every machine. Both POCs produce real numbers.

## Phase 1 — Day 2: Foundation: Corpus Ingestion & Baseline (Dense-Only) Retrieval
**Goal:** Text in, chunked and indexed, dense-only retrieval, naive LLM answer out.
- Finalize chunking (Doc_ID / §Section metadata).
- Ingest dev corpus into Qdrant (dense vectors).
- Build FastAPI skeleton `api/main.py` with one synchronous `/turn` endpoint.
- Hand-write 15 example queries (the initial eval set).
**Gate 2:** `/turn` answers 15 queries without crashing. ≥ 10/15 retrieve correct `Doc_ID`.

## Phase 2 — Day 3: Hybrid Retrieval, Fusion & Reranking
**Goal:** Dense + sparse hybrid, RRF fusion, cross-encoder rerank, dedup.
- Add BM25 sparse vector, calibrate `avg_len`.
- Implement hybrid query `prefetch(dense) + prefetch(sparse)` with RRF.
- Add cross-encoder reranking.
- Ablation #1: Compare hybrid+rerank against dense-only.
**Gate 3:** Hybrid query returns fused, deduped, reranked results. Ablation #1 has real numbers.

## Phase 3 — Day 4: Streaming Controller & Early Retrieval (-> G2)
**Goal:** Build incremental chunk simulator + controller deciding Wait/Retrieve/Suppress.
- Build `streaming/stream_simulator.py`.
- Implement Retrieval Controller (rule-based heuristics + LLM classifier).
- Extend eval set with presentation-only and single simple question cases.
- Compute G2 (Early Retrieval) live.
**Gate 4:** G2 ≥ 80% on eval set. False-trigger rate measured. Log decisions with timestamps and reasons.

## Phase 4 — Day 5: Multi-Intent Decomposition & Parallel Retrieval (-> G3)
**Goal:** Split compound queries into orthogonal sub-queries, retrieve concurrently.
- Implement decomposer (structured LLM call, hard cap on sub-queries).
- Wire to Phase 2 hybrid retrieval using `asyncio.gather`.
- Merge/dedup per-sub-query results.
- Add ≥ 15 compound-utterance cases to eval set.
**Gate 5:** G3 (Multi-Intent Identification) ≥ 70% on compound cases. Low over-fragmentation on single-questions.

## Phase 5 — Day 6: Session-Aware Synthesis, Grounding & Refinement (-> G4, G5)
**Goal:** Grounded cited answer + session memory + refinement/suppression classification.
- Implement synthesis prompt (inline citations, uncertainty).
- Implement deterministic grounding validator (G4).
- Implement `session/store.py` (ephemeral, keyed by `session_id`).
- Implement refinement classifier (`NEW_TOPIC`, `LATE_DETAIL`, `PRESENTATION_ONLY`).
- Add ≥ 10 late-detail cases and ≥ 10 presentation-only cases.
**Gate 6:** G4 ≥ 85% citation support, zero fabricated IDs. G5: 100% of refinement/suppression cases behave correctly.

## Phase 6 — Day 7: Observability, Reproducibility, Demo UI & Full Gate Run (-> G1, G6)
**Goal:** Make system visible, lock reproducibility, run all six gates.
- Finalize `TelemetryEvent` schema (G6).
- Build minimal Demo UI.
- Finalize `docker-compose.yml`, `README.md`, `run_eval.sh`.
- Test full stack on clean machine.
**Gate 7:** Clean-machine boot completes with zero manual steps. G6 schema coverage is 100%. Full scorecard exists.

## Phase 7 — Day 8: Hardening, Ablations & Optimization
**Goal:** Clear Phase 6 scorecard punch list, complete architectural ablations, tune latency/cost.
- Triage Phase 6 scorecard.
- Finalize Ablation #1 (hybrid vs dense).
- Run Ablation #2 (rule-based vs model-based controller).
- Document ≥ 3 edge-case failures.
- Latency pass.
**Gate 8:** All 6 gates meet target. Ablations written up. Edge cases documented.

## Phase 8 — Day 9: Final Testing, Demo Recording & Submission
**Goal:** Freeze system, record demo, submit before deadline.
- Final clean-machine test.
- Record ≤ 5-minute demo video.
- Finalize Architecture Brief and Benchmark Report.
- Build submission PPT.
- Tag final commit: `PRISM_GENAI_HACKATHON_Y2026`.
- Submit via Google Form.
