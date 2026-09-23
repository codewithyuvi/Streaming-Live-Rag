# ADR-4: LLM Provider — Dual-Provider Architecture (Groq + Gemini)

**Status:** Accepted & Implemented (Audited September 23, 2026)

**Decision:** We implement a **Dual-Provider Architecture**:
1. **Groq (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`)** via centralized `llm_config.py`: Handles all latency-sensitive operations (Streaming Controller deciding `trigger_now` / `wait` / `no_retrieval_needed`, Multi-Intent Decomposer, and Session Refinement Classifier).
2. **Gemini (`gemini-2.5-flash` / `gemini-3.8-flash`)** via official `google-genai` SDK: Handles final Session-Aware Synthesis, complex reasoning, and grounded citation generation.

---

### Context & Justification
The Samsung PRISM GenAI Hackathon theme evaluates two competing dimensions:
- **Streaming Latency / Time-to-First-Token (TTFT)**
- **Citation Grounding & Synthesis Precision**

During Phase 0/1 testing, Gemini produced exceptional reasoning and adherence to citation constraints, but its end-to-end API turnaround (800ms–2500ms) is too slow for chunk-by-chunk streaming controller decisions, where each decision must complete in < 400ms to avoid freezing the audio stream simulator.

Groq's LPU (Language Processing Unit) hosting offers generation latencies of **~200ms–380ms**, making it ideal for the streaming controller loop. However, synthesizing multi-page policy citations with strict adherence to `[Doc_ID §Section]` formats benefits from Gemini Flash's long-context attention and calibrated hallucination resistance.

---

### Centralized Client & Resilience Architecture (C1, M4)
To prevent runtime crashes and handle transient cloud issues during judging:
- All Groq calls are centralized through `llm_config.call_fast()`.
- **Transient-Only Retries:** The wrapper retries only transient errors (`429`, `500`, `502`, `503`, timeouts) with exponential backoff and randomized jitter.
- **Fail Loudly:** Authentication, invalid model IDs, or schema mismatches fail immediately with descriptive errors rather than silently degrading.
- **Client Lazy Loading:** Clients are initialized on first invocation to prevent import-time crashes in test runners.

---

### Model Evolution & Fallback Matrix
- **Decommissioned:** `llama3-8b-8192` (decommissioned by Groq in 2025/2026).
- **Fast Controller Primary:** `llama-3.1-8b-instant` (fallback: `llama-3.3-70b-versatile`).
- **Synthesis Primary:** `gemini-2.5-flash` (fallback: `gemini-3.8-flash` / `gemini-2.0-flash`).

Both keys (`GROQ_API_KEY` and `GEMINI_API_KEY`) are managed in `.env` and loaded at runtime.

