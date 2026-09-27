# Streaming Live RAG: Operational Runbook

This runbook provides complete operational instructions for running, testing, evaluating, demonstrating, and deploying the Streaming Live RAG system across all completed phases (Phase 0 through Phase 6).

---

## 1. Prerequisites & Environment Setup

### 1.1 Dual API Key Configuration
The system uses a **Dual-Provider Architecture**:
- **Groq (`GROQ_API_KEY`)**: Powering the ultra-low latency Streaming Controller, Multi-Intent Decomposer, and Session Refinement classifier (~200–400ms).
- **Gemini (`GEMINI_API_KEY`)**: Powering final Session-Aware Synthesis and Citation Grounding.

Copy `.env.example` to `.env` and fill in your API keys (or configure them dynamically via the web UI BYOK settings):
```bash
cp .env.example .env
```
Key configuration parameters in `.env`:
```env
# Fast Provider (Groq)
GROQ_API_KEY=gsk_your_groq_api_key_here
FAST_LLM_MODEL=openai/gpt-oss-20b          # or llama-3.1-8b-instant

# Quality Provider (Gemini)
GEMINI_API_KEY=AIzaSy_your_gemini_api_key_here
SYNTHESIS_LLM_MODEL=gemini-3.5-flash-lite  # or gemini-2.5-flash

# Vector Database (Remote or Embedded Auto-Fallback)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=http://localhost:6333
```

---

### 1.2 Execution Modes

#### Option A: Direct Python (Recommended for Local Dev & Testing)
*No Docker required!* If remote Qdrant is unreachable on port 6333, the system automatically uses embedded local on-disk storage (`data/qdrant_storage`).

```bash
# 1. Activate your Python environment (Python 3.10+)
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FastAPI server with Uvicorn (auto-seeds corpus on startup)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option B: Docker Compose (Containerized Production Boot)
```bash
# Build and start Qdrant and FastAPI container
docker compose up --build -d

# Verify containers are healthy
docker compose ps
curl http://localhost:8000/health
```

#### Option C: Cloudflare Quick Tunnel (Public Web Demo)
To share the live interactive interface with external reviewers or mobile devices without opening firewall ports:
```bash
cloudflared tunnel --url http://localhost:8000
```
This produces an ephemeral public URL (e.g. `https://random-word.trycloudflare.com`) routing directly to your running instance.

---

## 2. API Transports & Contracts

### 2.1 HTTP Simulated Pacing Endpoint (`POST /turn`)
Replays an utterance with simulated speech pacing (default: 2 words per chunk every 300ms) to exercise the live controller:

```bash
curl -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess_demo_01",
    "turn_id": 1,
    "utterance": "What are the dimensions and capacity of the Pune venue?"
  }'
```

**Response Format:**
```json
{
  "answer": "The Pune venue accommodates up to 45 attendees in Classroom setup [Doc_01 §1].",
  "telemetry": {
    "session_id": "sess_demo_01",
    "turn_id": 1,
    "refinement_type": "NEW_TOPIC",
    "latencies_ms": {
      "controller": 240,
      "retrieval": 78,
      "synthesis": 810,
      "end_to_end": 1128
    },
    "grounding_score": 1.0
  },
  "thoughts": [
    {
      "type": "thought",
      "stage": "intent_detected",
      "t_s": 0.84,
      "thought": "Informational inquiry regarding venue dimensions and capacity detected."
    },
    {
      "type": "thought",
      "stage": "provisional_search",
      "t_s": 0.85,
      "thought": "Speculative retrieval triggered early for 'Pune venue dimensions capacity'."
    },
    {
      "type": "thought",
      "stage": "decomposition_planned",
      "t_s": 1.42,
      "thought": "Single intent verified upon utterance completion."
    },
    {
      "type": "thought",
      "stage": "synthesis_ready",
      "t_s": 2.25,
      "thought": "Response synthesized with 1 validated citation [Doc_01 §1]."
    }
  ]
}
```

---

### 2.2 Live WebSocket Endpoint (`/ws/stream`)
Full-duplex transport for real-time speech / incremental transcript streaming:

- **Client -> Server Messages:**
  - Start turn: `{"type": "start", "session_id": "sess_123"}`
  - Partial speech chunk: `{"type": "chunk", "text": "What is the capacity..."}`
  - Utterance finished: `{"type": "end"}`

- **Server -> Client Live Events:**
  - `stream_started`: `{"type": "stream_started", "session_id": "...", "t_s": 0.0}`
  - `transcript`: `{"type": "transcript", "partial_text": "...", "t_s": 0.3}`
  - `controller`: `{"type": "controller", "trigger": "trigger_now|wait", "reason": "...", "t_s": 0.6}`
  - `thought`: `{"type": "thought", "stage": "intent_detected|provisional_search|...", "thought": "..."}`
  - `retrieval`: `{"type": "retrieval", "trigger": "provisional|delta", "hits": [...]}`
  - `answer`: `{"type": "answer", "answer": "...", "citations": [...], "grounding_score": 1.0}`
  - `telemetry`: `{"type": "telemetry", "telemetry": {...}}`

---

### 2.3 BYOK & Provider Configuration (`/config/llm`)
Manage API keys and providers at runtime without service restarts:
- `GET /config/llm` — View currently active fast and synthesis models (keys are masked).
- `POST /config/llm` — Update model IDs, base URLs, or API keys:
  ```bash
  curl -X POST http://localhost:8000/config/llm \
    -H "Content-Type: application/json" \
    -d '{"fast_model": "llama-3.1-8b-instant", "synthesis_model": "gemini-3.5-flash-lite"}'
  ```
- `POST /config/llm/test` — Ping configured providers to test authentication and measure turnaround latency:
  ```bash
  curl -X POST http://localhost:8000/config/llm/test
  ```

---

## 3. Automated Testing Suite

The codebase includes comprehensive unit tests verifying the pipeline, session store, hybrid search, grounding validator, and the 4-phase thought stream engine.

```bash
# Run all unit tests
pytest tests/ -v

# Run thought stream specific tests
pytest tests/test_thought_stream.py -v

# Run integration tests
pytest tests/test_pipeline.py -v
```

---

## 4. Phase-by-Phase Execution Guide

### Phase 0: Infrastructure & POC Validation (Gate 1)
Validate connectivity to Qdrant vector database and dual LLM providers:
```bash
python scripts/poc_a_qdrant.py
python scripts/poc_b_llm.py
```

### Phase 1 & 2: Ingestion & Hybrid Search Pipeline
Ingest corpus documents with both Dense (BGE-Small) and Sparse (BM25 with `Modifier.IDF`) embeddings:
```bash
python retrieval/ingest.py
```

### Phase 3: Streaming Controller & Early Retrieval (Gate 2)
The controller evaluates chunks incrementally. If the query stabilizes before the user stops speaking, it fires provisional retrieval at $t_1$:
```bash
python eval/gates/g2_early_retrieval.py
```

### Phase 4: Multi-Intent Decomposition (Gate 3)
Compound utterances are decomposed into 1..4 orthogonal sub-queries, retrieved in parallel, and merged via Quota Merge:
```bash
python eval/gates/g3_multi_intent.py
```

### Phase 5: Session Refinement & Grounding (Gates 4 & 5)
Handles multi-turn conversational context with strict answer versioning ($1 \to 1 \to 2$) and claim-level grounding verification:
```bash
python eval/gates/g4_grounding.py
python eval/gates/g5_session_refinement.py
```

---

## 5. Master Evaluation Harness (Gates G1–G6)

Run all 6 evaluation gates with a single command to generate official scorecard files:

- **Cross-Platform Python:**
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

Outputs:
- Terminal scorecard display with pass/fail metrics.
- Machine-readable JSON: `eval/results/scorecard.json`.
- Markdown report: `eval/results/scorecard.md`.

---

## 6. Troubleshooting & FAQ

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Remote Qdrant unreachable` | Docker Qdrant not running on `:6333` | The system automatically falls back to embedded storage in `data/qdrant_storage`. No action needed unless remote Qdrant is strictly desired (`docker compose up -d qdrant`). |
| `GROQ_API_KEY is not set` | Missing key in `.env` | Add key to `.env` or click "Provider Settings" in the Web UI to input your key via BYOK modal. |
| `FastEmbed download timeout` | Slow network on initial run | Pre-download model: `python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"` |
| `UI shows API Offline` | Uvicorn server is not running | Start the server with `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`. The UI also features a built-in simulated mode for offline walkthroughs. |
| `ADMIN_TOKEN forbidden` | Non-localhost caller hitting admin routes | Set `ADMIN_TOKEN` in `.env` and pass `X-Admin-Token` header, or access directly via `localhost:8000`. |

---

## 7. Manual QA Checklist

Before judging or demo submission, run these verifications against `http://localhost:8000`:

1. **Idle Load:** Load the page and wait 2+ minutes. Ensure zero phantom turns appear.
2. **Typing Isolation:** Type in the chat box without pressing Enter. Verify Send remains clickable and no turn opens.
3. **Preset Scenarios:** Click each preset button (Early Retrieval, Multi-Intent, Chit-Chat, Refinement) and verify the 4-phase thought stream badges populate dynamically.
4. **Hard Timeout Protection:** Active turns have a client-side 60s hard ceiling with automatic AbortController fallback.
5. **BYOK Live Ping:** Click "Test Connection" in the Settings modal to verify model latencies return real milliseconds rather than errors.
6. **Session Reset:** Click "Reset Session" and confirm that version ($v=1$), latencies, and grounding scores reset cleanly.
