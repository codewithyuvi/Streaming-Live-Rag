# ADR-4: LLM Provider — Dual-Provider Architecture with Dynamic BYOK Configuration

**Status:** Accepted, Implemented & Extended (September 2026)

**Decision:** We implement a **Dual-Provider Architecture** supplemented with dynamic runtime **BYOK (Bring Your Own Key)** configuration:
1. **Fast Controller Provider (Groq / OpenAI-compatible / Ollama):** Default model `openai/gpt-oss-20b` or `llama-3.1-8b-instant` via centralized `llm_config.py`. Handles all latency-sensitive operations (Streaming Controller deciding `trigger_now` / `wait` / `no_retrieval_needed`, Multi-Intent Decomposer, and Session Refinement Classifier) at sub-400ms turnaround.
2. **Quality Synthesis Provider (Gemini / Claude / OpenAI):** Default model `gemini-3.5-flash-lite` or `gemini-2.5-flash` via official `google-genai` SDK. Handles final Session-Aware Synthesis, complex reasoning, and claim-level citation generation.
3. **Dynamic BYOK & Provider Switching (`/config/llm`):** Providers, models, base URLs, and API keys can be updated at runtime without restarting the server. Sensitive API keys are treated as write-only and are strictly masked in read endpoints.

---

### Context & Justification
The Samsung PRISM GenAI Hackathon theme evaluates two competing dimensions:
- **Streaming Latency / Time-to-First-Token (TTFT)**
- **Citation Grounding & Synthesis Precision**

During Phase 0/1 testing, Gemini produced exceptional reasoning and adherence to citation constraints, but its end-to-end API turnaround (800ms–2500ms) is too slow for chunk-by-chunk streaming controller decisions, where each decision must complete in < 400ms to avoid freezing the audio stream simulator.

Groq's LPU (Language Processing Unit) hosting offers generation latencies of **~200ms–380ms**, making it ideal for the streaming controller loop. However, synthesizing multi-page policy citations with strict adherence to `[Doc_ID §Section]` formats benefits from Gemini Flash's long-context attention and calibrated hallucination resistance.

---

### Centralized Client & Resilience Architecture
To prevent runtime crashes and handle transient cloud issues during judging:
- All fast calls route through `llm_config.call_fast()`.
- **Transient-Only Retries:** The wrapper retries only transient errors (`429`, `500`, `502`, `503`, timeouts) with exponential backoff and randomized jitter.
- **Fail Loudly:** Authentication, invalid model IDs, or schema mismatches fail immediately with descriptive errors rather than silently degrading.
- **Client Lazy Loading:** Clients are initialized on first invocation to prevent import-time crashes in test runners.
- **Connection Test Endpoint (`POST /config/llm/test`):** Allows users and reviewers to ping configured providers directly from the UI or terminal to verify credentials and measure live round-trip latency.

---

### Model Evolution & Fallback Matrix
- **Fast Controller Primary:** `openai/gpt-oss-20b` / `llama-3.1-8b-instant` (fallbacks: `llama-3.3-70b-versatile`, local Ollama).
- **Synthesis Primary:** `gemini-3.5-flash-lite` / `gemini-2.5-flash` (fallbacks: `gemini-3.8-flash`, `gemini-2.0-flash`).
