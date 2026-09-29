# 9-Day Execution Plan & Deliverables

This document tracks the phased day-by-day execution plan for the Streaming Live RAG project (Samsung PRISM GenAI Hackathon 2026-27, Theme 4).

> **Current Status:** ✅ **100% COMPLETED ACROSS ALL PHASES (PHASE 0 THROUGH PHASE 8)**

---

## Phase 0 — Day 1: Technology Research, Architecture Lock & Foundation
**Goal:** Lock tech stack, stand up skeleton repo, Docker, shared schemas.
- [x] Complete 5 ADRs (ADR-1 through ADR-5).
- [x] Execute POC A (Qdrant hybrid round-trip) & POC B (LLM provider smoke test).
- [x] Scaffold repo, define Pydantic schemas in `telemetry/schema.py`.
- [x] Authored initial dev corpus (`venue_booking.txt`, `travel_policy.txt`).
- [x] Initialized `docs/RISKS.md`.
**Gate 1 (informal):** Boot clean, POCs produce real numbers. (✅ **Passed**)

---

## Phase 1 — Day 2: Foundation: Corpus Ingestion & Baseline Retrieval
**Goal:** Text in, chunked and indexed, dense-only retrieval, naive LLM answer out.
- [x] Finalize chunking with deterministic `Doc_ID` and `§Section` metadata.
- [x] Ingest dev corpus into Qdrant (`retrieval/ingest.py`).
- [x] Build FastAPI skeleton in `api/main.py` with `/turn` endpoint.
- [x] Authored initial 15 benchmark queries in `eval/labeled_set.yaml`.
**Gate 2:** `/turn` answers queries without crashing; retrieves correct `Doc_ID`. (✅ **Passed**)

---

## Phase 2 — Day 3: Hybrid Retrieval, Fusion & Reranking
**Goal:** Dense + sparse hybrid, RRF fusion, cross-encoder rerank, dedup.
- [x] Add BM25 sparse vector with `Modifier.IDF`.
- [x] Implement hybrid query `prefetch(dense) + prefetch(sparse)` with Reciprocal Rank Fusion (RRF).
- [x] Add FastEmbed `TextCrossEncoder` (`ms-marco-MiniLM-L-6-v2`) reranking.
- [x] Completed Ablation #1 (Hybrid+Rerank vs. Dense-Only) in `docs/ablations/hybrid_vs_dense.md`.
**Gate 3:** Hybrid query returns fused, deduped, reranked results. (✅ **Passed**)

---

## Phase 3 — Day 4: Streaming Controller & Early Retrieval (Gate G2)
**Goal:** Build incremental chunk stream + controller deciding Wait / Retrieve / Suppress.
- [x] Built live streaming queue and playback in `streaming/live_stream.py`.
- [x] Implement Two-Stage Retrieval Controller (rule-based stop-word heuristics + fast LLM classifier).
- [x] Measured G2 Early Retrieval Trigger rate live (100% on eligible queries, 0% false triggers on chit-chat).
- [x] Log decisions with timestamps and trigger reasons.
**Gate 4:** G2 ≥ 80% on eval set with zero false-trigger rate. (✅ **Passed**)

---

## Phase 4 — Day 5: Multi-Intent Decomposition & Parallel Retrieval (Gate G3)
**Goal:** Split compound queries into orthogonal sub-queries, retrieve concurrently.
- [x] Implement decomposer in `controller/decompose.py` (structured LLM call with quota bounds).
- [x] Execute parallel hybrid retrieval across sub-queries using `asyncio.gather`.
- [x] Implement Quota Result Merge (`retrieval/merge.py`: min 2 chunks per sub-query, cap 8).
- [x] Added compound-utterance benchmark cases to `eval/labeled_set.yaml`.
**Gate 5:** G3 Multi-Intent Identification ≥ 70% on compound cases. (✅ **Passed**)

---

## Phase 5 — Day 6: Session-Aware Synthesis, Grounding & Refinement (Gates G4, G5)
**Goal:** Grounded cited answer + session memory + refinement/suppression classification.
- [x] Implement session-aware synthesis prompt with citation enforcement.
- [x] Implement deterministic Claim-Level Grounding Validator (`retrieval/grounding.py`).
- [x] Implement ephemeral in-memory session store (`session/store.py`) with answer version lineage ($1 \to 1 \to 2 \to 1$) and citation unioning.
- [x] Implement refinement classifier (`NEW_TOPIC`, `LATE_DETAIL`, `PRESENTATION_ONLY`).
- [x] Verified G4 (100% citation support, 0 fabricated IDs) and G5 (refinement and search suppression).
**Gate 6:** G4 ≥ 85% citation support, zero fabricated IDs; G5 verified. (✅ **Passed**)

---

## Phase 6 — Day 7: Observability, Reproducibility, Demo UI & Scorecard (Gates G1, G6)
**Goal:** Make system visible, lock reproducibility, run all six gates.
- [x] Finalize `TelemetryEvent` schema (`telemetry/schema.py`) and JSONL persistence sink.
- [x] Build interactive demonstration dashboard in `static/index.html`.
- [x] Finalize `docker-compose.yml`, `requirements.txt`, `run_eval.sh`, and `run_eval.bat`.
- [x] Master evaluation runner (`eval/run_eval.py`) generates `scorecard.json` and `scorecard.md`.
**Gate 7:** Clean-machine boot completes; G6 telemetry schema coverage is 100%. (✅ **Passed**)

---

## Phase 7 — Day 8: Engine Unification & 4-Phase Live Thought Stream
**Goal:** Single source of truth for all transports, live explainability, automated test suite.
- [x] Unified turn pipeline into `streaming/engine.py` supporting both `/ws/stream` and `/turn`.
- [x] Emitted real-time 4-phase honest thought stream (`intent_detected`, `provisional_search`, `decomposition_planned`, `synthesis_ready`).
- [x] Added HTTP replay collector (`TurnResponse.thoughts`) for REST clients.
- [x] Built comprehensive unit test suite in `tests/test_thought_stream.py` and `tests/test_pipeline.py` (22/22 tests passing).
- [x] Added embedded Qdrant local storage fallback (`data/qdrant_storage`) for zero-Docker execution.
- [x] Finalized Ablation #1 (Hybrid vs Dense) and Ablation #2 (Heuristics vs Model-Based Controller) in `docs/BENCHMARK_REPORT.md`.
**Gate 8:** All gates pass; thought stream verified; 22 unit tests green. (✅ **Passed**)

---

## Phase 8 — Day 9: Final Hardening, Documentation & Submission Freeze
**Goal:** Freeze system, refresh complete documentation suite, prepare release.
- [x] Verified clean execution on localhost port 8000 and Cloudflare tunnel.
- [x] Authored top-level root `README.md` and updated `streaming-live-rag/README.md`.
- [x] Refreshed `docs/RUNBOOK.md`, `docs/ARCHITECTURE_BRIEF.md`, `docs/CHECKLIST.md`, and `docs/RISKS.md`.
- [x] Synchronized `agents.md` collaboration log.
- [x] Committed to git `main` branch and verified GitHub remote synchronization.
**Gate 9:** Release tagged and ready for submission: `PRISM_GENAI_HACKATHON_Y2026`. (✅ **Passed**)

---

## Phase 9: Continuous Speculative Streaming, Live VS Arena, GPU Acceleration & Voice
**Goal:** Next-generation real-time multi-intent speculation, live microphone voice input, side-by-side arena, and GPU acceleration.
- [x] Continuous simultaneous multi-intent speculative streaming in `streaming/engine.py` (detects multiple intent clauses mid-stream, fires up to 4 parallel searches before utterance ends).
- [x] Collective coverage checking (70% token/stem match) suppressing redundant delta searches upon speech finish.
- [x] Live Side-by-Side VS Arena (`/ws/compare` and `/ws/dual_stream`, `POST /turn/compare`) comparing Streaming Live RAG vs. Naive Sequential RAG concurrently with session isolation and cache bypass.
- [x] Synchronized browser voice streaming via Web Speech API with real-time waveform visualizer.
- [x] Hardware GPU acceleration (NVIDIA CUDA 12) dropping hybrid retrieval latency from ~170ms to **~9.8ms** (BGE-Small ~3ms, Cross-Encoder ~3ms).
- [x] High-performance query response cache (`_QUERY_CACHE`) with instant Phase 0 thought streaming.
- [x] Strict startup corpus isolation (`Doc_01` and `Doc_02` strictly on boot) with dynamic user uploads saved to `data/uploads/`.
- [x] Expanded test suite to 26 unit tests in `tests/test_pipeline.py` (100% passing).
**Gate 10:** Complete live voice + GPU acceleration + VS Arena verified. All 26 tests green. (✅ **Passed**)

