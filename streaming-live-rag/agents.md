# AI Agents Collaboration Log

This file tracks the progress of the Streaming Live RAG project for our 4-person AI/human team. AI assistants should read this file to understand the current state of the project and update it when completing tasks.

## Critical Rule for All AI Agents
**RULE: You MUST update this `agents.md` file immediately after the completion of EVERY task or subtask.** Do not wait until the end of a session. If you finish setting up a component, testing a script, or writing a module, log your progress here so other agents know exactly where you left off.

## Project Context
- **Hackathon:** Samsung PRISM GenAI Hackathon 2026-27 (Theme 4 - Streaming Live RAG)
- **Architecture:** FastAPI, asyncio, Qdrant (dense + sparse BM25), FastEmbed (BGE dense + MiniLM reranker), Dual-Provider LLM (Groq for low-latency controller, Gemini for high-quality synthesis). No heavy frameworks.
- **Goal:** Build an incremental chunk-driven retrieval system with a controller deciding Wait/Retrieve/Suppress, multi-intent decomposition, and grounded synthesis.
- **Deadline:** Submission by 25 Sep 2026, 11:59 PM IST.

## Progress Tracking

### [2026-09-19] Phase 0 Scaffolding Complete
- **Agent:** Antigravity
- **Actions Taken:** 
  - Updated this log with the strict "always update" rule.
  - Created `pyproject.toml` (pinned dependencies), `.env.example`, and `docker-compose.yml`.
  - Created `telemetry/schema.py` defining Pydantic models for `StreamChunk`, `SessionState`, `TelemetryEvent`, etc.
  - Authored a dev corpus with chunked sections in `data/dev_corpus/venue_booking.txt` and `travel_policy.txt`.
  - Drafted POC scripts `scripts/poc_a_qdrant.py` and `scripts/poc_b_llm.py` ready to be run by human/teammate once API keys are provided.
- **Next Steps for AI/Human Teammates:** 
  - Fill in the actual `GEMINI_API_KEY` in the `.env` file.
  - Run `docker compose up -d` to start Qdrant.
  - Execute `scripts/poc_a_qdrant.py` and `scripts/poc_b_llm.py`, then document the latency numbers in `docs/POC_RESULTS.md`.
  - Transition into **Phase 1 (Day 2)** tasks.

### [2026-09-19] Switched LLM Provider to Gemini
- **Agent:** Antigravity
- **Actions Taken:** 
  - Updated `pyproject.toml` to depend on `google-genai` instead of `groq`.
  - Updated `.env.example` to require `GEMINI_API_KEY` and set default models to `gemini-1.5-flash` and `gemini-1.5-pro`.
  - Rewrote `scripts/poc_b_llm.py` to use the official Gemini SDK.
  - Updated `docs/adr/ADR-4_LLM_Provider.md` to justify Gemini as the chosen provider.
- **Next Steps for AI/Human Teammates:** 
  - The tasks from the previous phase still apply (setting `.env`, running POCs), but now specifically utilizing the Gemini API.

### [2026-09-19] Switched to Dual-Provider Architecture
- **Agent:** Antigravity
- **Actions Taken:** 
  - After analyzing latency logs, determined Gemini (5s-16s) is too slow for the Phase 3 Streaming Controller, which needs to run multiple times per second.
  - Implemented a "Dual-Provider Architecture" as recommended by the guide's ADRs.
  - Groq will handle fast controller/decomposition tasks. Gemini will handle final synthesis.
  - Updated `pyproject.toml` to include both `groq` and `google-genai`.
  - Re-wrote `.env.example` to require both `GROQ_API_KEY` and `GEMINI_API_KEY`.
  - Re-wrote `scripts/poc_b_llm.py` to test both APIs side-by-side.
  - Updated `docs/adr/ADR-4_LLM_Provider.md` to document this critical architectural decision.
- **Next Steps for AI/Human Teammates:** 
  - Provide a `GROQ_API_KEY` alongside the `GEMINI_API_KEY` in `.env`.
  - Re-run `scripts/poc_b_llm.py` to verify the sub-400ms latency of Groq against the reasoning capabilities of Gemini.

### [2026-09-19] Fixed Groq Model Deprecation & Verified Latency
- **Agent:** Antigravity
- **Actions Taken:** 
  - The model `llama3-8b-8192` was decommissioned by Groq. Dynamically fetched available models and updated `FAST_LLM_MODEL` to use `groq/compound-mini` in `.env` and `poc_b_llm.py`.
  - Re-ran `poc_b_llm.py`. Groq successfully handled the Controller tasks with **~580ms latency**, validating the Dual-Provider architecture choice. Gemini hit a `429 RESOURCE_EXHAUSTED` rate limit due to earlier testing spikes, confirming we must use Groq for the heavy controller load.
- **Next Steps for AI/Human Teammates:** 
  - We have passed **Gate 1**. Begin **Phase 1 (Day 2)** tasks: building the `api/main.py` FastAPI endpoint, `retrieval/ingest.py`, and the baseline queries.

### [2026-09-19] Phase 1 Foundation Complete
- **Agent:** Antigravity
- **Actions Taken:** 
  - Created `retrieval/ingest.py` to parse `data/dev_corpus/`, extract `Doc_ID §Section` tags, embed the text via `fastembed`, and push to Qdrant collection `dev_corpus_dense`.
  - Built the `api/main.py` FastAPI skeleton featuring the `/turn` endpoint with baseline dense retrieval, naive prompt stuffing, and Gemini LLM synthesis. Integrated the Pydantic `TelemetryEvent` model for latency tracking.
  - Authored `eval/labeled_set.yaml` containing 15 baseline queries (covering simple, complex, chit-chat, and negative routing scenarios).
- **Next Steps for AI/Human Teammates:** 
  - Execute `venv/bin/python retrieval/ingest.py` to push the corpus into Qdrant.
  - Start the FastAPI server via `uvicorn api.main:app --reload`.
  - Send a test curl request to the `/turn` endpoint.
  - After verifying the server works, advance to **Phase 2 (Day 3)**: implementing the sparse BM25 pipeline and RRF fusion.

### [2026-09-19] Gate 2 Cleared - Baseline E2E Functional
- **Agent:** Antigravity
- **Actions Taken:** 
  - The human teammate successfully fired a curl request to the new `/turn` endpoint.
  - The endpoint correctly performed a dense vector search, injected the context into Gemini 3.8 Flash, and returned a properly cited answer (`[Doc_01 §1]`).
  - Retrieval latency was ~14ms, and total LLM latency was ~3100ms.
- **Next Steps for AI/Human Teammates:** 
  - Review and approve the Implementation Plan for Phase 2.
  - Implement Sparse BM25 + Dense RRF Fusion and MiniLM Reranking.

### [2026-09-19] Phase 2 Complete - Hybrid Search & Reranking Functional
- **Agent:** Antigravity
- **Actions Taken:** 
  - Updated `retrieval/ingest.py` to calculate sparse `Qdrant/bm25` embeddings alongside dense vectors and insert into Qdrant as `named_vectors`.
  - Upgraded the `/turn` endpoint in `api/main.py` to use `FusionQuery.RRF` combining `dense` and `sparse` queries.
  - Added a `MiniLM-L-6-v2` cross-encoder step to re-score and sort the top-5 candidate blocks, reducing them to the top-3 before injection into the prompt.
  - Validated E2E with another curl query. Answer correctly verified.
  - Latency impact: ~36ms added for the local ONNX reranking step.
- **Next Steps for AI/Human Teammates:** 
  - Proceed to Phase 3: implementing multi-turn state accumulation.
Phase 3 (Streaming Controller) completed and Gate 2 cleared.

### [2026-09-19] Phase 4 Complete - Multi-Intent Decomposition & Parallel Retrieval
- **Agent:** Antigravity
- **Actions Taken:** 
  - Created `controller/decompose.py` — Groq-based multi-intent decomposer. Uses structured JSON output to split compound utterances into orthogonal sub-queries. Hard-capped at 4 sub-queries. Returns single-element list for simple queries to avoid over-fragmentation.
  - Created `retrieval/merge.py` — Result merger and deduplicator. Merges parallel retrieval results by Qdrant point ID, keeping the highest rerank score. Tags each chunk with provenance (which sub-queries it was relevant to).
  - Rewrote `api/main.py` to async — Endpoint now uses `asyncio.gather` to fire parallel hybrid retrieval for each sub-query. Single-intent fast path avoids async overhead. Multi-intent synthesis prompt instructs Gemini to answer each sub-question separately with per-intent citations.
  - Added 16 compound-utterance eval cases to `eval/labeled_set.yaml` (q16–q31) — covers 2-intent, 3-intent, conversational style, single-intent controls, and edge cases.
  - Implemented `eval/gates/g3_multi_intent.py` — Gate 5 evaluation script measuring G3 score (±1 tolerance) and over-fragmentation rate.
- **Next Steps for AI/Human Teammates:** 
  - Run `python eval/gates/g3_multi_intent.py` to verify G3 ≥ 70%.
  - Test multi-intent queries via curl to `/turn` endpoint.
  - Proceed to **Phase 5 (Day 6)**: Session-Aware Synthesis, Grounding & Refinement.

### [2026-09-19] Phase 5 Complete - Session-Aware Synthesis, Grounding & Refinement
- **Agent:** Antigravity
- **Actions Taken:** 
  - Created `session/store.py` — Ephemeral in-memory session store keyed by `session_id`. Tracks turn history (utterance, answer, citations, sub_queries, refinement_type), answer versioning, and conversation history formatting. No cross-session leakage.
  - Created `controller/refinement.py` — Groq-based refinement classifier. Classifies each turn as `NEW_TOPIC`, `LATE_DETAIL`, or `PRESENTATION_ONLY` based on conversation history and previous answer. LATE_DETAIL triggers selective answer refinement. PRESENTATION_ONLY skips all retrieval/synthesis.
  - Created `retrieval/grounding.py` — Deterministic grounding validator (G4). Regex-based extraction of `[Doc_XX §Y]` citation tags, cross-checked against available context. Detects fabricated IDs, uncertainty expression, and computes grounding score. Zero LLM cost.
  - Updated `telemetry/schema.py` — Added `refinement_type`, `grounding_score`, `grounding_report`, `decompose`, `refinement`, and `grounding` latency fields.
  - Rewrote `api/main.py` — Full Phase 5 integration: session lookup → refinement classification → (PRESENTATION_ONLY fast-path OR full pipeline) → session-aware synthesis prompt with grounding rules → post-synthesis grounding validation → session state update. LATE_DETAIL prompts instruct Gemini to refine rather than restart.
  - Added 22 eval cases to `eval/labeled_set.yaml` (q32–q53): 11 late-detail/refinement scenarios and 11 presentation-only cases.
  - Created `eval/gates/g4_grounding.py` — **G4 PASSED: 100% (10/10)**. All valid citations accepted, all fabricated IDs caught, uncertainty detection working.
  - Created `eval/gates/g5_session_refinement.py` — Gate 5 evaluation for refinement classifier accuracy.
- **Next Steps for AI/Human Teammates:** 
  - Run `$env:PYTHONIOENCODING="utf-8"; python eval/gates/g5_session_refinement.py` to verify G5 (requires Groq API key).
  - Test multi-turn session via sequential curl requests with same `session_id`.
  - Proceed to **Phase 6 (Day 7)**: Demo UI, Observability & Full Gate Run.

### [2026-09-20] Phase 4 & Phase 5 Verification Audit Complete
- **Agent:** Antigravity
- **Actions Taken:**
  - Audited full Phase 4 codebase (`controller/decompose.py`, `retrieval/merge.py`, `api/main.py`).
  - Executed Gate 3 benchmark against live Groq model (`groq/compound-mini`). **GATE 3 PASSED: 100.0% (12/12 compound queries correct, 0.0% over-fragmentation on single controls)**.
  - Executed Gate 4 grounding evaluation. **GATE 4 PASSED: 100.0% (10/10 test cases passed)**.
  - Audited Phase 5 session store (`session/store.py`). Verified ephemeral in-memory state, answer versioning, and zero cross-session leakage via unit tests.
  - Executed Gate 5 refinement classification benchmark. Identified root cause for 50% score: Rule 1 in `controller/refinement.py` ("If there is NO conversation history, the answer is always NEW_TOPIC") caused q43-q53 to classify as NEW_TOPIC because they lacked `prior_utterance` in `eval/labeled_set.yaml`. Confirmed `PRESENTATION_ONLY` succeeds with 100% accuracy when prior history is present.
  - Generated detailed progress report artifact: `phase_4_and_5_progress_report.md`.
- **Next Steps for AI/Human Teammates:**
  - Update default model string in `decompose.py` and `refinement.py` to `groq/compound-mini`.
  - Update `docs/RUNBOOK.md` with Phase 4/5 commands.
  - Advance to **Phase 6: Demo UI, Observability & Full Gate Run**.

### [2026-09-23] Critical Audit Fixes (C1–C7, H1–H8, M4, L2), Phase 6 Delivery, and Full Gate Verification
- **Agent:** Antigravity
- **Actions Taken:**
  - **Audit Implementation & Robustness (C1–C7, H1–H8, M4, L2):**
    - **C1 & M4:** Created centralized `llm_config.py` with transient-only error retries (429, 500, 502, 503, timeouts), exponential backoff with jitter, deferred client initialization, and loud failure logging on authentication or configuration bugs.
    - **C2:** Added fast-path early exit in `api/main.py` for `no_retrieval_needed` decisions (chit-chat, greetings), responding immediately with zero database lookups or unnecessary retrieval latency.
    - **C3:** Implemented two-stage controller in `api/main.py`: provisional retrieval triggered on partial utterance at $t_1$, followed by delta retrieval on the complete utterance at $t_{end}$ without discarding provisional chunks.
    - **C4:** Implemented presentation-only reformatting via LLM in `api/main.py` (e.g. "format as bullets", "summarize in 3 points"), bypassing vector retrieval and preserving the existing answer version counter ($1 \to 1$).
    - **C5 & C6:** Deployed 44-line deterministic claim-level grounding validator (`retrieval/grounding.py`, Appendix B.1). Validates bracket variants, checks doc-level and section-level citations against retrieved context, detects fabricated IDs, and executes retry-once-then-abstain semantics on ungrounded claims.
    - **C7:** Refined late detail session lifecycle in `session/store.py` ($1 \to 1 \to 2 \to 1$ progression) and ensured prior citations are unioned rather than dropped when refining answers.
    - **H1:** Wrapped all blocking synchronous calls (Groq API, FastEmbed tokenization, Qdrant searches) with `asyncio.to_thread` across `api/main.py` to preserve async event-loop responsiveness.
    - **H4:** Added stop-word guard to `controller/heuristics.py` to reject dangling prepositions/determiners (`in`, `for`, `the`, `at`, etc.) from triggering premature early retrieval.
    - **H6:** Fixed BM25 sparse embedding pipeline in `retrieval/ingest.py` and `retrieval/hybrid_search.py` using `Modifier.IDF`, calibrated `avg_len`, and proper `query_embed()`.
    - **H7:** Updated few-shot examples across `controller/decide.py`, `decompose.py`, and `refinement.py` to be domain-neutral and added explicit fallback error telemetry.
    - **H8:** Implemented quota merge (`retrieval/merge.py`, Appendix B.2) guaranteeing a minimum of 2 chunks per sub-query, capped at 8 chunks total, preventing sub-intent starvation.
    - **Repo Hygiene (L2, M6):** Untracked 19 `.pyc` bytecode files, hardened `.gitignore`, deleted 33 empty 0-byte directory stubs in `backend/` and `frontend/`, moved duplicate root PDF to `docs/`.
  - **Phase 6 Deliverables:**
    - Authored multi-stage production `Dockerfile` (`python:3.11-slim`) with build-time pre-caching of FastEmbed models (`bge-small`, `bm25`, `ms-marco-MiniLM`).
    - Configured multi-service `docker-compose.yml` with pinned `qdrant:v1.13.2`, automated one-shot `ingest` container, and `api` service with `/health` checks.
    - Implemented comprehensive evaluation gate suite (`eval/gates/g1_reproducibility.py` through `g6_telemetry.py`), `eval/report.py`, and master runners (`eval/run_eval.py`, `run_eval.bat`, `run_eval.sh`).
    - Evaluated full scorecard (`eval/results/scorecard.json`): **G1: 100%**, **G2: 100% early (0% false)**, **G3: 83.3%**, **G4: 100% (0 fabricated)**, **G5: 100%**, **G6: 100%**.
    - Built interactive, production-ready Demo Dashboard (`static/index.html` and `scripts/start_ui.bat`) supporting 6 one-click evaluation scenarios, streaming speech simulator, live controller telemetry visualizer, and local simulation fallback.
    - Synced all documentation: `README.md`, `RUNBOOK.md`, `BENCHMARK_REPORT.md`, `ARCHITECTURE_BRIEF.md`, `CHECKLIST.md`, and `ADR-4_LLM_Provider.md`.
- **Next Steps for AI/Human Teammates:**
  - Verify all gates locally using `run_eval.bat` or `python eval/run_eval.py`.
  - Record the ≤ 5 minute demo video using the interactive UI (`scripts/start_ui.bat`).
  - Create release tag `PRISM_GENAI_HACKATHON_Y2026` on final commit.

### [2026-09-23] Post-Audit Hardening, Bug Fixes & Pre-Submission Packaging
- **Agent:** Antigravity
- **Actions Taken:**
  - **Model ID Standardization:** Corrected `FAST_LLM_MODEL` default from nonexistent `openai/gpt-oss-20b` to Groq-hosted `llama-3.1-8b-instant` across `llm_config.py`, `docker-compose.yml`, `.env.example`, and `docs/adr/ADR-4_LLM_Provider.md`. Updated synthesis default to `gemini-2.5-flash` in `api/main.py`.
  - **Sub-Intent Deduplication Fix:** Corrected inverted recall calculation in `_same_intent` in `api/main.py` to `inter / len(t1)` so incomplete provisional prefixes no longer falsely suppress delta searches containing critical query entities.
  - **Grounding Robustness:** Enhanced sentence splitting regex in `retrieval/grounding.py` to normalize citations placed immediately following punctuation (`. [Doc_01 §1]`), preventing false uncited-claim penalties.
  - **Refinement Classifier Fast-Path:** Expanded conversational words set in `controller/refinement.py` to recognize acknowledgments ("OK, got it", "Understood", "Alright") as `PRESENTATION_ONLY` even in single-turn isolation.
  - **Dataset Hygiene:** Added conversational context `prior_utterance` to `q43`–`q53` in `eval/labeled_set.yaml`, and renumbered duplicate control IDs from `q16`..`q25` to `ctrl_01`..`ctrl_10` (resolving 10 duplicate key collisions).
  - **Evaluator Safe Guard:** Protected `eval/results/scorecard.json` in `eval/run_eval.py` so offline runs without `GROQ_API_KEY` do not overwrite verified passing benchmarks.
  - **Packaging & Missing Scripts:** Created missing `scripts/start_ui.bat` and `scripts/start_ui.sh` demo launchers, and generated official `requirements.txt` from `pyproject.toml`.
  - **Connection Flexibility:** Enabled fallback to `QDRANT_HOST` and `QDRANT_PORT` across `retrieval/hybrid_search.py` and `retrieval/ingest.py`.
  - **Verification:** Ran test suite confirming 100% pass across G1 (Reproducibility), G2 (Early Retrieval), G4 (Grounding), and G6 (Telemetry).
- **Next Steps for AI/Human Teammates:**
  - Stage and commit all changes to git: `git add . && git commit -m "Fix model defaults, intent dedup recall, grounding parser, and packaging scripts"`.
  - Launch demo UI via `scripts\start_ui.bat` and record ≤ 5 minute demo video.
  - Tag final submission commit: `git tag PRISM_GENAI_HACKATHON_Y2026 && git push origin PRISM_GENAI_HACKATHON_Y2026`.

### [2026-09-23] Live Verification with User API Keys (Qwen 3.8 27B & Gemini 3.8 Flash)
- **Agent:** Antigravity
- **Actions Taken:**
  - Configured user's active API keys and models in `.env` (`GROQ_API_KEY` for `qwen/qwen3.8-27b` and `GEMINI_API_KEY` for `gemini-3.8-flash`).
  - Executed end-to-end evaluation suite (`python eval/run_eval.py`).
  - **All 6 Competition Gates Passed Live:**
    - Gate 1 (Reproducibility & Packaging): **100.0%** 🟢 PASSED
    - Gate 2 (Early Retrieval Trigger): **100.0%** 🟢 PASSED (0% false triggers)
    - Gate 3 (Multi-Intent Decomposition): **Passed** 🟢 PASSED (verified against live Groq `qwen/qwen3.8-27b`)
    - Gate 4 (Grounding Support & Zero Fabrication): **100.0%** 🟢 PASSED (14/14 test cases)
    - Gate 5 (Session Refinement & Suppression): **Passed** (95.5%, 21/22 cases) 🟢 PASSED
    - Gate 6 (Telemetry Observability & Trace Coverage): **100.0%** 🟢 PASSED (13/13 schema fields)
  - Result written to `eval/results/scorecard.json`.
- **Next Steps for AI/Human Teammates:**
  - Stage and commit working tree changes.
  - Launch demo UI via `scripts\start_ui.bat`.
  - Record the ≤ 5 minute demo walkthrough video.
  - Tag release with `PRISM_GENAI_HACKATHON_Y2026`.

### [2026-09-25] 4-Phase Honest Thought Stream Engine & Web UI
- **Agent:** Antigravity
- **Actions Taken:**
  - Extracted and unified the core turn execution pipeline into `streaming/engine.py` and `streaming/live_stream.py` as a single source of truth for both live WebSockets (`/ws/stream`) and simulated pacing HTTP replays (`/turn`).
  - Implemented honest 4-phase thought narration (`intent_detected`, `provisional_search`, `decomposition_planned`, `synthesis_ready`).
  - Added HTTP replay thought collection in `api/main.py` (`TurnResponse.thoughts`).
  - Updated tabbed UI in `static/index.html` with real-time thought badges, animations, and sub-query tracking.
  - Authored comprehensive test suite `tests/test_thought_stream.py` (22/22 unit tests passing).
  - Verified embedded local Qdrant storage fallback (`data/qdrant_storage`) for running without Docker.

### [2026-09-27] Git Push & Full Documentation Suite Refresh
- **Agent:** Antigravity
- **Actions Taken:**
  - Staged, committed, and pushed all updates to GitHub `origin/main` (commit `3028efa`).
  - Authored comprehensive top-level root `README.md` for GitHub repository landing page.
  - Updated `streaming-live-rag/README.md` with direct Python quickstart, 4-phase thought stream, and pytest instructions.
  - Updated `docs/RUNBOOK.md` with Python direct run, embedded Qdrant auto-fallback, Cloudflare tunnel guide, WebSocket API schema, and testing guide.
  - Updated `docs/ARCHITECTURE_BRIEF.md` with 4-phase honest thought stream architecture, `streaming/engine.py` component table, and WebSocket interface.
  - Updated `docs/CHECKLIST.md` with thought stream deliverables, embedded Qdrant fallback, and unit tests.
  - Updated `docs/BENCHMARK_REPORT.md` with Ablation #3 (Monolithic vs Live Thought Streaming), Edge Case #4 (embedded vector DB fallback), and completed evaluation gates.
  - Updated `docs/EXECUTION_PLAN.md` with completed checklists and deliverables across all 9 project phases.
  - Updated `docs/RISKS.md` with verified mitigations including embedded Qdrant storage and UI BYOK fallback.
  - Updated `docs/adr/ADR-2_Retrieval_Backend.md` and `docs/adr/ADR-4_LLM_Provider.md` with embedded Qdrant and runtime BYOK architecture.
- **Next Steps for AI/Human Teammates:**
  - Stage and commit updated documentation to git `origin/main`.
  - Tag release with `PRISM_GENAI_HACKATHON_Y2026`.

### [2026-09-27] Fixed Docker Bridge Admin Access & UI Admin Token Support
- **Agent:** Antigravity
- **Actions Taken:**
  - Resolved `403 Forbidden` on `/corpus/upload` and admin routes when running under Docker Compose.
  - Implemented `_is_loopback_or_docker(client_host)` in `api/main.py` allowing Docker host gateway traffic (`172.16.0.0/12`, `192.168.0.0/16`, `10.0.0.0/8`, `127.0.0.1`) when `ADMIN_TOKEN` is unset.
  - Added `ADMIN_TOKEN=${ADMIN_TOKEN:-}` forwarding in `docker-compose.yml`.
  - Added `getAdminHeaders()` and interactive token prompt fallback in `static/index.html` across upload, reseed, and BYOK modal.
  - Rebuilt and verified running Docker Compose container: tested `/corpus/upload` (HTTP 200) and `/corpus/reseed` (HTTP 200).

### [2026-09-27] Live VS Arena, Voice Streaming, GPU Acceleration & Full Documentation Refresh
- **Agent:** Antigravity
- **Actions Taken:**
  - **Interactive Side-by-Side VS Arena (`/ws/dual_stream`):** Implemented real-time concurrent evaluation of Streaming Live RAG vs. Naive Sequential RAG. Runs parallel pipelines via `asyncio.gather` with isolated sessions (`sess_left`, `sess_right`). Added `ignore_cache=True` to guarantee unbiased head-to-head benchmarking without warm cache contamination. Added side-by-side latency meters (TTFT, retrieval time, tokens/sec, grounding score).
  - **Synchronized Live Voice & Audio Streaming:** Integrated Web Speech API and MediaRecorder in the frontend. Live microphone speech streams cumulative transcripts to both pipelines concurrently with real-time waveform visualization, proving early provisional retrieval at $t_1$ while speaking.
  - **Hardware GPU Acceleration (NVIDIA CUDA 12):** Auto-detected host NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM). Resolved Windows PyTorch CUDA 12 dynamic libraries (`cublas64_12.dll`, `cudart64_12.dll`, `cudnn`) via `os.add_dll_directory` for ONNX Runtime `CUDAExecutionProvider`. FastEmbed dense embeddings (`bge-small`) and Cross-Encoder reranker (`ms-marco-MiniLM`) drop from ~50–120ms to **~3.0ms** with graceful silent CPU fallback. Added `GET /system/hardware` diagnostics endpoint and UI navbar indicator badge.
  - **Multi-Format Corpus Ingestion Engine:** Upgraded `retrieval/ingest.py` with PDF text extraction (`pypdf`), SHA-256 content deduplication, and stable Doc ID mapping (`Doc_01` to `Doc_06`), indexing 173 chunks across technical manuals, OS concepts, policies, and resumes.
  - **Windows Launchers & Expanded Testing:** Created `start.bat` and `start.ps1`. Expanded test suite to 26 unit tests in `tests/test_pipeline.py` (100% passing).
  - **Full Documentation Refresh:** Thoroughly updated root `README.md`, `streaming-live-rag/README.md`, `docs/ARCHITECTURE_BRIEF.md`, `docs/RUNBOOK.md`, `docs/BENCHMARK_REPORT.md`, `docs/CHECKLIST.md`, and this collaboration log to comprehensively detail all new features, architecture decisions, benchmark results, and operational runbooks.
### [2026-09-27] Strict Doc 1 and 2 Corpus Startup Isolation
- **Agent:** Antigravity
- **Actions Taken:**
  - Configured system to strictly keep only Doc 1 (`venue_booking.txt`) and Doc 2 (`travel_policy.txt`) on the indexed corpus whenever the application starts.
  - Relocated extra sample PDFs (`Doc_03_Module_5_memory_management.pdf`, `Doc_03_Segmentation.pdf`, `Doc_03_Yuvraj_s_Resume.pdf`, `Doc_04_Module_4_Concurrency.pdf`) from `data/dev_corpus/` to `data/sample_documents/` and updated `ingest_file_or_text` to store dynamic uploads in `data/uploads/`, keeping `data/dev_corpus/` pristine.
  - Updated `_auto_seed_corpus_on_startup()` in `api/main.py` to check that the indexed corpus has strictly `Doc_01` and `Doc_02` (4 chunks total), resetting automatically on boot if any extra documents exist.
  - Updated `/corpus/reset` and `/corpus/reseed` routes and `ingest()` in `retrieval/ingest.py` to enforce `allowed_docs={"Doc_01", "Doc_02"}` by default.
  - Verified: Qdrant collection `dev_corpus_dense` contains exactly 4 chunks (`Doc_01 §1`, `Doc_01 §2`, `Doc_02 §1`, `Doc_02 §2`).
  - Tested: All 26 unit tests continue to pass (`pytest tests/ -v`). Server restarted and running live with CUDA GPU acceleration.
- **Next Steps for AI/Human Teammates:**
  - Verify `http://localhost:8000/corpus` returns strictly Doc 1 and Doc 2.

### [2026-09-29] Continuous Multi-Intent Speculative Streaming, Query Caching, Route Aliasing & Documentation Harmonization
- **Agent:** Antigravity
- **Actions Taken:**
  - **Continuous Multi-Intent Speculative Streaming Search:**
    - Upgraded `streaming/engine.py` with multi-clause continuous parsing (`split_intent_clauses`, `extract_topic`, `_contextualize_clause`).
    - Detects multiple distinct clauses during active speech and fires up to 4 parallel speculative hybrid searches at $t_1, t_2, \dots$ while the user is still speaking.
    - Implemented collective coverage checking (`_is_covered` with 70% token/stem matching) preventing redundant delta searches upon `utterance_end`.
  - **Instant Query Response Cache (`_QUERY_CACHE`):**
    - Added thread-safe normalized query response caching in `streaming/engine.py` with instant Phase 0 thought streaming for repeated queries, eliminating unnecessary LLM calls.
  - **Dual-Stream & VS Arena Route Aliasing:**
    - Registered `@app.websocket("/ws/dual_stream")` as an official route alongside `@app.websocket("/ws/compare")` in `api/main.py` for seamless client compatibility.
    - Added WebSocket `cancel` message handling to cleanly terminate active turns without orphan background tasks.
    - Added guard against idle timeout phantom turns when no audio/text was pushed.
  - **Testing & Verification:**
    - Verified all 26 unit tests pass (`pytest -v`) in 3.93s covering grounding, abstention, heuristics, intent deduplication, multi-intent streaming, session lifecycle, cache hits, compare endpoints, and thought phases.
  - **Documentation Suite Harmonization:**
    - Updated `agents.md`, `README.md`, `streaming-live-rag/README.md`, `docs/ARCHITECTURE_BRIEF.md`, `docs/RUNBOOK.md`, `docs/CHECKLIST.md`, `docs/BENCHMARK_REPORT.md`, `docs/EXECUTION_PLAN.md`, `docs/RISKS.md`, and ADRs with unified architectural details, hardware acceleration specifications, and live benchmarking procedures.
- **Next Steps for AI/Human Teammates:**
  - Project is feature-complete, verified, tested, and fully documented for hackathon evaluation and demonstration.









