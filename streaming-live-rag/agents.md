# AI Agents Collaboration Log

This file tracks the progress of the Streaming Live RAG project for our 4-person AI/human team. AI assistants should read this file to understand the current state of the project and update it when completing tasks.

## Critical Rule for All AI Agents
**RULE: You MUST update this `agents.md` file immediately after the completion of EVERY task or subtask.** Do not wait until the end of a session. If you finish setting up a component, testing a script, or writing a module, log your progress here so other agents know exactly where you left off.

## Project Context
- **Hackathon:** Samsung PRISM GenAI Hackathon 2026-27 (Theme 4 - Streaming Live RAG)
- **Architecture:** FastAPI, asyncio, Qdrant (dense + sparse BM25), FastEmbed (BGE dense + MiniLM reranker), Gemini (LLM). No heavy frameworks like LangChain/LangGraph.
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

### [2026-09-19] POC Tests Executed
- **Agent:** Antigravity
- **Actions Taken:** 
  - Ran `docker compose up -d` to start the Qdrant database.
  - Executed `scripts/poc_a_qdrant.py` — successfully connected to Qdrant and created the required collection.
  - Executed `scripts/poc_b_llm.py` — confirmed that `gemini-1.5-flash` was deprecated via the SDK, and automatically updated all configurations to use `gemini-3.8-flash`. 
  - Logged test latency results (~3500ms TTFT) and JSON parsing success in `docs/POC_RESULTS.md`. Note: We experienced some `503 High Demand` errors, so we will need exponential backoff logic for the final benchmark runs.
- **Next Steps for AI/Human Teammates:** 
  - We have fully passed **Gate 1**! We are now ready to begin **Phase 1 (Day 2)** tasks: building the `api/main.py` FastApi endpoint, `retrieval/ingest.py`, and writing the 15 baseline queries.
