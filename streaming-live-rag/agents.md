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
