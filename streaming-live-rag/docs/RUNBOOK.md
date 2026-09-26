# Streaming Live RAG: Project Runbook

This runbook provides complete operational instructions for running, testing, evaluating, and demonstrating the Streaming Live RAG system across all completed phases (Phase 0 through Phase 6).

---

## 1. Prerequisites & Environment Setup

### 1.1 Dual API Key Configuration
The system uses a **Dual-Provider Architecture**:
- **Groq (`GROQ_API_KEY`)**: Powering the ultra-low latency Streaming Controller, Multi-Intent Decomposer, and Session Refinement classifier (~200–400ms).
- **Gemini (`GEMINI_API_KEY`)**: Powering final Session-Aware Synthesis and Citation Grounding.

Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Ensure `.env` contains:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=AIzaSy_your_gemini_api_key_here
FAST_LLM_MODEL=llama-3.1-8b-instant        # or qwen/qwen3.8-27b
SYNTHESIS_LLM_MODEL=gemini-2.5-flash       # or gemini-1.5-flash
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 1.2 Execution Modes

#### Option A: Docker Compose (Single-Command Production Run)
Requirements: Docker & Docker Compose.
```bash
# Build and start Qdrant, run auto-ingest, and start FastAPI
docker compose up --build -d

# Verify services are healthy
docker compose ps
curl http://127.0.0.1:8000/health
```

#### Option B: Local Python Environment (Recommended for Development & Grading)
Requirements: Python 3.11+.
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Phase-by-Phase Execution Guide

### Phase 0: Infrastructure & POC Validation (Gate 1)
Validate connectivity to Qdrant vector database and dual LLM providers:
```bash
# Test Qdrant connectivity and vector operations
python scripts/poc_a_qdrant.py

# Test Groq vs Gemini dual-provider latency and connectivity
python scripts/poc_b_llm.py
```

---

### Phase 1 & 2: Ingestion & Hybrid Search Pipeline
Ingest corpus documents with both Dense (BGE-Small) and Sparse (BM25 with `Modifier.IDF`) embeddings, then start the FastAPI service:

```bash
# Step 1: Ingest the corpus into Qdrant
python retrieval/ingest.py

# Step 2: Start the FastAPI pipeline server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify turn endpoint:**
```bash
curl -X POST http://127.0.0.1:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_01", "turn_id": 1, "utterance": "What are the dimensions and capacity of the Pune venue?"}'
```

---

### Phase 3: Streaming Controller & Early Retrieval (Gate 2)
The controller evaluates chunks incrementally. If the query stabilizes before the user stops speaking, it fires provisional retrieval at $t_1$, followed by delta retrieval at $t_{end}$:

```bash
# Test early triggering with trailing hesitation
curl -X POST http://127.0.0.1:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_02", "turn_id": 1, "utterance": "I was wondering, what is the maximum capacity of the Pune venue because I have a lot of guests coming?"}'
```
*Inspect telemetry JSON: `retrieval_events` shows early trigger at $t_1$, and `controller_decisions` logs the trigger rationale.*

**Run Gate 2 benchmark:**
```bash
python eval/gates/g2_early_retrieval.py
```

---

### Phase 4: Multi-Intent Decomposition (Gate 3)
Compound utterances (e.g. asking about both capacity AND cancellation policies) are decomposed into 1..N orthogonal sub-queries, retrieved in parallel, and merged via Quota Merge (min 2 chunks per sub-query, cap 8):

```bash
# Test multi-intent query
curl -X POST http://127.0.0.1:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_03", "turn_id": 1, "utterance": "What is the Pune hall capacity, and what is the refund policy if we cancel 10 days before?"}'
```

**Run Gate 3 benchmark:**
```bash
python eval/gates/g3_multi_intent.py
```

---

### Phase 5: Session Refinement, Grounding & Presentation (Gates 4 & 5)
Handles multi-turn conversational context with strict answer versioning:
1. **Turn 1 (NEW_TOPIC):** Fetches context, synthesizes answer ($v=1$).
2. **Turn 2 (PRESENTATION_ONLY):** "Summarize that in 3 bullet points" — bypasses retrieval, reformats via LLM, keeps version ($v=1$).
3. **Turn 3 (LATE_DETAIL):** "What if there are 200 guests?" — retrieves delta context, unions prior citations, updates version ($v=2$).
4. **Claim-Level Grounding:** Enforces deterministic citation verification (`[Doc_ID §Section]`), zero fabricated IDs, and retry-once-then-abstain semantics.

**Run Gate 4 & Gate 5 benchmarks:**
```bash
# Gate 4: Grounding and citation validation
python eval/gates/g4_grounding.py

# Gate 5: Session refinement and continuity
python eval/gates/g5_session_refinement.py
```

---

## 3. Phase 6: Interactive Demo UI & Master Evaluation

### 3.1 Interactive Demo Dashboard
Launch the web interface to visually simulate streaming speech, observe early retrieval triggers, inspect decomposed sub-queries, and verify citation grounding in real time:

- **Windows One-Click Launcher:**
  ```cmd
  scripts\start_ui.bat
  ```
- **Linux / macOS:**
  ```bash
  chmod +x scripts/start_ui.sh
  ./scripts/start_ui.sh
  ```
- **Or via Browser:**
  Open `static/index.html` directly in any web browser, or navigate to `http://localhost:8000/demo` while the API server is running.

**Preset Scenarios available in Demo UI:**
1. Simple Question with Early Retrieval (`q01`)
2. Compound Multi-Intent Query (`q16`)
3. Chit-Chat / Suppression (Zero DB queries) (`q12`)
4. Multi-Turn Late Detail Refinement ($1 \to 1 \to 2$) (`q32`)
5. Presentation-Only Reformatting (No search) (`q44`)
6. Out-of-Corpus Uncovered Query (`q14`)

---

### 3.2 Master Evaluation Harness (Gates G1–G6)
Run all 6 evaluation gates with a single command to generate the scorecard:

- **Command Line (Cross-Platform):**
  ```bash
  python eval/run_eval.py
  ```
- **Windows Batch Script:**
  ```cmd
  run_eval.bat
  ```
- **Unix Shell Script:**
  ```bash
  ./run_eval.sh
  ```

**Results Output:**
- Terminal scorecard display with pass/fail thresholds.
- JSON output: `eval/results/scorecard.json`.
- Markdown report: `eval/results/scorecard.md`.

---

## 4. Troubleshooting & FAQ

| Issue | Resolution |
| :--- | :--- |
| `GROQ_API_KEY is not set` | Ensure `.env` exists in the `streaming-live-rag` directory with a valid Groq API key. |
| `Qdrant connection refused` | Run `docker compose up -d qdrant` or start local Qdrant on port 6333. |
| `FastEmbed download timeout` | Pre-download models or run `python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"` |
| `UI shows API Offline` | The UI includes an automatic fallback to local simulated mode so you can test all 6 scenarios even without a running backend. To connect live, start `uvicorn api.main:app --reload`. |

---

## 5. Manual QA Checklist (Post-Fix Verification)

The items below cover behavior that is UI/timing-driven and cannot run under
`pytest` (in particular, the client-side 60s turn-timeout clock, which is
plain browser JS with no test runner wired into this repo). Run these by
hand against `uvicorn api.main:app --reload` on `:8000` after any change to
`static/index.html`, `api/main.py`, or `streaming/engine.py`:

1. **No phantom turns on idle load.** Load the page fresh and wait 2+
   minutes without touching it. Zero "Thinking…" bubbles should appear.
2. **Typing never opens a turn.** Type a query character by character and
   never press Enter/Send. No turn opens, Send stays clickable, the input
   stays editable.
3. **One turn per real action.** Run both benchmark scenarios plus one typed
   query ended with Enter. Each opens exactly one turn, and the clock stops
   on the real answer.
4. **60s hard timeout (test (d)).** Manually stall a turn past 60s — or
   temporarily lower `TURN_TIMEOUT_MS` in `static/index.html` (WS path) and
   the `setTimeout(..., TURN_TIMEOUT_MS)` in `submitUserMessage()`'s
   `AbortController` (HTTP-fallback path) to something small, e.g. 5000, to
   test faster. Confirm: the "Thinking (Ns)…" clock/pill stops updating, the
   bubble shows the "exceeded the 60s limit" message, and Send/input
   re-enable. This is the one check that genuinely can't run under pytest
   since the clock is client-side `setInterval`/`setTimeout` — if a JS test
   runner is ever added to this repo, promote this to an automated test.
5. **Test Connection shows real names.** Click Test Connection in the BYOK
   modal → real provider/model names (or a real error string), never the
   literal text `undefined`.
6. **Reseed/upload failure is not disguised as success.** With `ADMIN_TOKEN`
   unset, call Reseed/Upload from a non-loopback origin (e.g. through a
   tunnel/proxy) → an error alert is shown, not a false "success" message
   with `undefined` interpolated into it.
7. **HUD fully clears on session reset.** Run a turn so the HUD populates,
   then reset the session mid-conversation → latency, grounding, intents,
   and version all clear back to their placeholder values (`— ms`, `—`,
   `0 Intents`, `v1`), not just the version.


