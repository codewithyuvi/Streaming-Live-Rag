# ⚡ Streaming Live RAG — Theme 4
**Samsung PRISM GenAI Hackathon 2026–27**  
*A low-latency, session-aware retrieval-augmented generation pipeline with real-time speech controller, 4-phase honest thought stream, multi-intent decomposition, and deterministic grounding verification.*

---

## 🌟 Overview & Key Innovations

Streaming Live RAG addresses the fundamental challenge in real-time conversational voice and text systems: **latency vs. factual accuracy**. Conventional RAG systems wait until the user finishes speaking, run a monolithic vector search, and then prompt an LLM.

Our pipeline changes this paradigm:
1. **Speculative Early Retrieval (Gate G2)**: Evaluates partial speech transcripts chunk-by-chunk (every ~300ms) with a trailing stop-word stability guard. Launches provisional hybrid retrieval in the background while the user is still speaking.
2. **4-Phase Honest Thought Stream**: Emits structured, timestamped cognitive events in real-time across both WebSocket streaming (`/ws/stream`) and pacing-simulated HTTP turns (`/turn`):
   - 🔍 `intent_detected`: Real-time intent classification mid-utterance.
   - ⚡ `provisional_search`: Early speculative dense + sparse retrieval.
   - 🧩 `decomposition_planned`: Multi-intent query breakdown on compound utterances.
   - 🛡️ `synthesis_ready`: Grounded answer synthesis with citation validation and confidence scoring.
3. **Multi-Intent Query Decomposition & Quota Merge (Gate G3)**: Breaks compound questions into 1..4 orthogonal sub-queries, executes parallel searches, and guarantees evidence quota (min 2 chunks per sub-query).
4. **Deterministic Claim-Level Grounding (Gate G4)**: A strict 44-line deterministic validator eliminates fabricated citations and verifies claim support before emission.
5. **Session-Aware Refinement & Version Lineage (Gate G5)**: Manages state across conversational turns (`NEW_TOPIC` resets to $v=1$, `LATE_DETAIL` increments to $v=2$ with citation unioning, and `PRESENTATION_ONLY` reformats directly with 0 database lookups).
6. **Zero-Dependency Embedded Fallback**: Automatically falls back from remote Qdrant to local on-disk storage (`data/qdrant_storage`), allowing the entire pipeline to run natively without requiring Docker.

---

## 🚀 Quickstart

You can run Streaming Live RAG via **Direct Python** (recommended for local development) or **Docker Compose**.

### Option 1: Direct Python (Fastest, No Docker Required)

```bash
# 1. Navigate to the project directory
cd streaming-live-rag

# 2. Set up environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY and GEMINI_API_KEY (optional: BYOK in the UI)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server (auto-seeds corpus into local embedded Qdrant on first boot)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, open:
- 🌐 **Interactive Dashboard:** [http://localhost:8000](http://localhost:8000) or [http://localhost:8000/demo](http://localhost:8000/demo)
- 🩺 **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- 📑 **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Docker Compose (Single-Command Production Run)

```bash
cd streaming-live-rag
cp .env.example .env
docker compose up --build -d
```

### Option 3: Expose via Cloudflare Tunnel

To share a live interactive demo link:
```bash
cloudflared tunnel --url http://localhost:8000
```
This generates a public `.trycloudflare.com` URL that tunnels directly to your local instance.

---

## 🧪 Testing & Evaluation

### Run Unit Tests
The test suite verifies the pipeline, hybrid search, grounding validator, and the 4-phase thought stream engine:
```bash
cd streaming-live-rag
pytest tests/ -v
# 22 passed in ~3s
```

### Run Evaluation Gate Harness (Gates G1–G6)
```bash
cd streaming-live-rag
python eval/run_eval.py    # Cross-platform
./run_eval.sh              # Linux / macOS
.\run_eval.bat             # Windows
```

Scorecards are saved to `eval/results/scorecard.json` and `eval/results/scorecard.md`.

---

## 📊 Evaluation Benchmark Scorecard

| Gate | Name | Target | Measured | Result |
| :--- | :--- | :--- | :--- | :---: |
| **G1** | **Reproducibility & Packaging** | Single-command clean boot, 100% | **100.0%** (5/5) | 🟢 **PASSED** |
| **G2** | **Early Retrieval Trigger** | $\ge 80\%$ eligible queries, 0% false triggers | **100.0%** | 🟢 **PASSED** |
| **G3** | **Multi-Intent Decomposition** | $\ge 70\%$ compound queries decomposed | **Passed** | 🟢 **PASSED** |
| **G4** | **Grounding Validation** | $\ge 85\%$ support, 0 fabricated IDs | **100.0%** (14/14) | 🟢 **PASSED** |
| **G5** | **Session Refinement** | $\ge 90\%$ refinement & suppression | **95.5%** (21/22) | 🟢 **PASSED** |
| **G6** | **Telemetry Observability** | 100% trace coverage & persistence | **100.0%** (25/25) | 🟢 **PASSED** |

---

## 🏗️ Architecture & Pipeline Flow

```
User Audio / ASR Stream (300ms chunks)
                │
                ▼
   ┌────────────────────────────────────────┐
   │    Streaming Live Engine (G2)          │
   │  - Stability & stop-word guard         │
   │  - Speculative early retrieval (t1)    │
   │  - Thought Phase 1: intent_detected    │
   │  - Thought Phase 2: provisional_search │
   └───────────────────┬────────────────────┘
                       │
             Utterance End Clock (t_end)
                       │
                       ▼
   ┌────────────────────────────────────────┐
   │   Refinement & Multi-Intent (G3)       │
   │  - NEW_TOPIC / LATE_DETAIL             │
   │  - PRESENTATION_ONLY (bypass DB)       │
   │  - 1..4 orthogonal sub-queries         │
   │  - Thought Phase 3: decomposition_plan │
   └───────────────────┬────────────────────┘
                       │
                       ▼
   ┌────────────────────────────────────────┐
   │   Hybrid Search & Quota Merge          │
   │  - FastEmbed BGE Dense (384d)          │
   │  - Qdrant BM25 Sparse (Modifier.IDF)   │
   │  - Quota Merge (min 2 chunks / intent) │
   │  - Cross-Encoder Reranker              │
   └───────────────────┬────────────────────┘
                       │
                       ▼
   ┌────────────────────────────────────────┐
   │   Grounded Synthesis & G4              │
   │  - Gemini Flash synthesis              │
   │  - Deterministic regex citation check  │
   │  - Retry-once-then-abstain logic       │
   │  - Thought Phase 4: synthesis_ready    │
   └───────────────────┬────────────────────┘
                       │
                       ▼
   Streamed Answer + Citations + Telemetry (G6)
```

---

## 🔌 API Transports & Endpoints

### 1. HTTP Turn Endpoint: `POST /turn`
Replays speech input with pacing simulation (words-per-chunk and millisecond intervals) and returns telemetry along with the 4-phase thought history:
```bash
curl -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo_session",
    "turn_id": 1,
    "utterance": "What are the dimensions and capacity of the Pune venue?"
  }'
```

**Response Format:**
```json
{
  "answer": "The Pune venue accommodates up to 45 attendees in Classroom setup [Doc_01 §1].",
  "telemetry": {
    "session_id": "demo_session",
    "turn_id": 1,
    "latencies_ms": {"controller": 240, "retrieval": 78, "synthesis": 810, "end_to_end": 1128},
    "grounding_score": 1.0
  },
  "thoughts": [
    {"type": "thought", "stage": "intent_detected", "t_s": 0.84, "thought": "Informational query regarding venue capacity."},
    {"type": "thought", "stage": "provisional_search", "t_s": 0.85, "thought": "Fired provisional retrieval for 'Pune venue capacity'."},
    {"type": "thought", "stage": "decomposition_planned", "t_s": 1.42, "thought": "Single intent confirmed at utterance end."},
    {"type": "thought", "stage": "synthesis_ready", "t_s": 2.25, "thought": "Grounded answer synthesized with 1 validated citation."}
  ]
}
```

### 2. Live WebSocket Endpoint: `ws://localhost:8000/ws/stream`
Full-duplex real-time streaming for live speech chunks (microphone / live ASR typing):
- **Client sends:**
  - `{"type": "start", "session_id": "abc123"}`
  - `{"type": "chunk", "text": "cumulative speech..."}`
  - `{"type": "end"}`
- **Server emits in real time:**
  - `{"type": "transcript", ...}`
  - `{"type": "controller", ...}`
  - `{"type": "retrieval_started", "trigger": "provisional", ...}`
  - `{"type": "thought", "stage": "intent_detected|provisional_search|...", ...}`
  - `{"type": "answer", ...}`
  - `{"type": "telemetry", ...}`

### 3. Dynamic BYOK Configuration: `/config/llm`
Update LLM providers, models, or API keys at runtime without restarting the server:
- `GET /config/llm`: Inspect active providers and models (write-only keys are masked).
- `POST /config/llm`: Update credentials/models.
- `POST /config/llm/test`: Ping providers to measure live latency.

---

## 📁 Repository Structure

```
Streaming-Live-Rag/
├── README.md                      # Primary repository landing page & documentation
└── streaming-live-rag/
    ├── api/
    │   └── main.py                # FastAPI routes, rate limiter, & CORS
    ├── controller/
    │   ├── decide.py              # Two-stage controller (Wait / Retrieve / Suppress)
    │   ├── decompose.py           # Multi-intent query decomposer
    │   ├── heuristics.py          # Trailing stop-word & stability guards
    │   └── refinement.py          # Session turn classifier (New / Late / Presentation)
    ├── data/
    │   ├── dev_corpus/            # Reference benchmark corpus documents
    │   └── qdrant_storage/        # Embedded local vector DB (auto-fallback)
    ├── docs/
    │   ├── ARCHITECTURE_BRIEF.md  # Detailed architecture & telemetry contract
    │   ├── BENCHMARK_REPORT.md    # Gate scores, ablations, & edge cases
    │   ├── CHECKLIST.md           # Submission requirements & verification matrix
    │   ├── RUNBOOK.md             # Operational runbook across all phases
    │   └── adr/                   # Architecture Decision Records (ADR-1 to ADR-5)
    ├── eval/
    │   ├── dataset_loader.py      # Dataset parser & validator
    │   ├── labeled_set.yaml       # 64 benchmark queries & ground truth
    │   ├── report.py              # Scorecard formatter
    │   ├── run_eval.py            # Master evaluation harness
    │   └── gates/                 # Gate verification scripts (g1 to g6)
    ├── retrieval/
    │   ├── grounding.py           # Claim-level deterministic validator (ADR-5)
    │   ├── hybrid_search.py       # Dense + BM25 sparse RRF fusion & reranker
    │   ├── ingest.py              # Corpus chunking & Qdrant indexer
    │   ├── merge.py               # Multi-intent quota merger
    │   └── parsers.py             # Document format parsers
    ├── session/
    │   └── store.py               # In-memory session state & version progression
    ├── static/
    │   └── index.html             # Interactive demo dashboard with live thoughts
    ├── streaming/
    │   ├── engine.py              # Unified transport-agnostic live turn engine
    │   └── live_stream.py         # Async live queue & pacing utterance playback
    ├── telemetry/
    │   ├── schema.py              # Strict Pydantic TelemetryEvent schema
    │   └── sink.py                # JSONL telemetry persistence sink
    ├── tests/
    │   ├── test_pipeline.py       # End-to-end integration test suite
    │   └── test_thought_stream.py # 4-phase honest thought stream unit tests
    ├── Dockerfile                 # Pinned container with cached FastEmbed models
    ├── docker-compose.yml         # Qdrant + FastAPI multi-container stack
    ├── pyproject.toml             # Project dependencies & tool configurations
    └── requirements.txt           # Pinned production requirements
```

---

## 🔒 Security & Guardrails

- **Zero Data Leakage:** Sessions are strictly ephemeral, keyed by isolated `session_id`, and automatically cleaned up.
- **Admin Access Control:** Constant-time `hmac.compare_digest` verification for administrative operations (`/config/llm`, `/corpus/clear`, `/upload`). Loops back strictly to `localhost` when `ADMIN_TOKEN` is unset.
- **Rate Limiting & DoS Protection:** Dual-bucket rate limiter enforcing 60 requests/minute per (IP, session) and 300 requests/minute per egress IP with automated TTL sweeps.
- **Corpus Grounding Isolation:** All answers are derived strictly from the indexed local corpus; hallucinations and fabricated citation tags are trapped by the deterministic grounding engine.
