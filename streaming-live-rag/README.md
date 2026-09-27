# ⚡ Streaming Live RAG — Theme 4
**Samsung PRISM GenAI Hackathon 2026–27**  
*A low-latency, session-aware retrieval-augmented generation pipeline with real-time speech controller, 4-phase honest thought stream, multi-intent decomposition, and deterministic grounding verification.*

---

## 🚀 3-Command Quickstart

### Option 1: Direct Python (Fastest, No Docker Required)

```bash
# 1. Configure environment keys
cp .env.example .env
# Add your GROQ_API_KEY and GEMINI_API_KEY in .env (or configure at runtime via UI)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Boot server (Auto-seeds corpus on startup using embedded local Qdrant storage)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Docker Compose (Gate G1 Reproducibility)

```bash
# 1. Configure environment keys
cp .env.example .env

# 2. Boot container stack (Qdrant + FastEmbed + FastAPI)
docker compose up --build -d

# 3. Run full benchmark evaluation (Gates G1–G6)
./run_eval.sh      # On Linux / macOS
.\run_eval.bat     # On Windows
```

Once running, access the **Interactive Live Demo UI** at:  
👉 **`http://localhost:8000`** (or `http://localhost:8000/demo`)  
🩺 Healthcheck: `http://localhost:8000/health`  
📑 Swagger Docs: `http://localhost:8000/docs`

### Expose via Cloudflare Tunnel
```bash
cloudflared tunnel --url http://localhost:8000
```

---

## 📊 Benchmark Scorecard

| Gate | Name | Target | Measured | Result |
| :--- | :--- | :--- | :--- | :---: |
| **G1** | **Reproducibility & Packaging** | Single-command clean boot, 100% | **100.0%** (5/5) | 🟢 **PASSED** |
| **G2** | **Early Retrieval Trigger** | $\ge 80\%$ eligible queries, 0% false triggers | **100.0%** | 🟢 **PASSED** |
| **G3** | **Multi-Intent Decomposition** | $\ge 70\%$ compound queries decomposed | **Passed** | 🟢 **PASSED** |
| **G4** | **Grounding Validation** | $\ge 85\%$ support, 0 fabricated IDs | **100.0%** (14/14) | 🟢 **PASSED** |
| **G5** | **Session Refinement** | $\ge 90\%$ refinement & suppression | **95.5%** (21/22) | 🟢 **PASSED** |
| **G6** | **Telemetry Observability** | 100% trace coverage & persistence | **100.0%** (25/25) | 🟢 **PASSED** |

Detailed scorecard results are saved to: `eval/results/scorecard.json` and `eval/results/scorecard.md`.

---

## 🧠 4-Phase Honest Thought Stream

Both the WebSocket stream (`/ws/stream`) and the HTTP turn endpoint (`/turn`) emit real-time structured thoughts reflecting honest internal execution state:

1. **Phase 1: `intent_detected`** — Speculatively classifies user intent while speech chunks arrive every ~300ms.
2. **Phase 2: `provisional_search`** — Fires pre-emptive hybrid retrieval (dense + sparse BM25) upon detecting a stable semantic prefix, before the user stops speaking ($t_1$).
3. **Phase 3: `decomposition_planned`** — At utterance completion ($t_{end}$), decomposes compound utterances into orthogonal sub-queries, launching delta retrieval if needed.
4. **Phase 4: `synthesis_ready`** — Synthesizes answers, verifies citation claims against indexed doc chunks, and calculates deterministic grounding score.

---

## 🏗️ Architecture & Pipeline Flow

```
User Audio / ASR Stream (300ms chunks)
                │
                ▼
   ┌────────────────────────────────────────┐
   │    Streaming Live Engine (G2)          │
   │  - Stability & stop-word guard         │
   │  - Early provisional search at t1      │
   │  - Chit-chat early return (0 DB)       │
   │  - Emits: intent_detected, prov_search │
   └───────────────────┬────────────────────┘
                       │
             Utterance Clock t_end
                       │
                       ▼
   ┌────────────────────────────────────────┐
   │   Refinement & Multi-Intent (G3)       │
   │  - NEW_TOPIC / LATE_DETAIL             │
   │  - PRESENTATION_ONLY (bypass DB)       │
   │  - 1..4 orthogonal sub-queries         │
   │  - Emits: decomposition_planned        │
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
   │  - Emits: synthesis_ready              │
   └───────────────────┬────────────────────┘
                       │
                       ▼
   Streamed Answer + Citations + Telemetry (G6)
```

---

## 🎯 Key Architectural Differentiators

1. **Two-Stage Controller (Resolving G2 vs G3)**:  
   When the controller triggers `retrieve_now`, it starts a background task immediately (`provisional` event at $t_1$) while **continuing to listen** to the incoming stream. At utterance end ($t_{end}$), it decomposes the full query and retrieves only the delta intents, merging all results seamlessly.

2. **Claim-Level Grounding Validator (ADR-5)**:  
   Deterministic regex validator that extracts citations across bracket formats (`[Doc_01 §1]`, `[Doc_01§1]`, combined brackets, parentheses) and computes sentence-level support. Eliminates fabricated IDs and enforces deterministic abstention on ungrounded claims.

3. **Quota Merging across Sub-Intents**:  
   Guarantees each sub-intent its top 2 evidence chunks so strong intents never starve weaker ones, ensuring full question coverage.

4. **Ephemeral Session Memory & Version Lineage**:  
   Clean version lifecycle (`1 -> 1 -> 2 -> 1`):
   - `NEW_TOPIC`: resets answer version to 1.
   - `LATE_DETAIL`: unions prior citations, contextualizes search query, and increments version.
   - `PRESENTATION_ONLY`: suppresses database search, reformats prior answer via LLM, and preserves version and state.

5. **Universal Embedded Fallback**:  
   If a remote Qdrant server is not running on port 6333, `retrieval/hybrid_search.py` automatically initializes local on-disk storage (`data/qdrant_storage`), allowing the entire application to run natively on any host without Docker.

---

## 🧪 Automated Testing

Run the integration and thought stream test suites:
```bash
pytest tests/ -v
```
All 22 unit tests verify:
- Stream chunking and pacing simulation
- 4-phase thought narration progression
- Early provisional triggering and delta retrieval merging
- Citation extraction and zero fabricated IDs
- Ephemeral session versioning and citation unioning

---

## 📁 Repository Structure

```
streaming-live-rag/
├── api/
│   └── main.py              # FastAPI live pipeline, rate limiter, & /turn endpoint
├── controller/
│   ├── decide.py            # Stream controller (Wait/Retrieve/Suppress)
│   ├── decompose.py         # Multi-intent query decomposer
│   ├── heuristics.py        # Stability heuristics & stop-word guards
│   └── refinement.py       # Refinement classifier (New/Late/Presentation)
├── data/
│   ├── dev_corpus/          # Ingestible reference documents
│   └── qdrant_storage/      # Embedded local vector DB storage
├── docs/
│   ├── ARCHITECTURE_BRIEF.md# Component interfaces, contracts, & telemetry
│   ├── BENCHMARK_REPORT.md  # Benchmark results, ablations, & edge cases
│   ├── CHECKLIST.md         # Submission requirements & verification matrix
│   ├── RUNBOOK.md           # Step-by-step operational runbook
│   └── adr/                 # Architecture Decision Records (ADR-1 to ADR-5)
├── eval/
│   ├── dataset_loader.py    # Robust dataset loader
│   ├── labeled_set.yaml     # 64 benchmark queries & ground truth
│   ├── report.py            # Scorecard formatter
│   ├── run_eval.py          # Master evaluation runner (G1-G6)
│   └── gates/               # Gate verification scripts (g1 to g6)
├── retrieval/
│   ├── grounding.py         # Claim-level grounding validator (ADR-5)
│   ├── hybrid_search.py     # FastEmbed dense + BM25 RRF & reranker
│   ├── ingest.py            # Corpus parser & Qdrant ingestion (IDF BM25)
│   ├── merge.py             # Sub-intent quota merger
│   └── parsers.py           # Document format parsers
├── session/
│   └── store.py             # Ephemeral in-memory session state
├── static/
│   └── index.html           # Interactive demo dashboard UI with live thoughts
├── streaming/
│   ├── engine.py            # Transport-agnostic live turn engine & thought streamer
│   └── live_stream.py       # Async live stream queue & pacing utterance playback
├── telemetry/
│   ├── schema.py            # TelemetryEvent Pydantic schema
│   └── sink.py              # JSONL persistence sink
├── tests/
│   ├── test_pipeline.py     # End-to-end integration tests
│   └── test_thought_stream.py# 4-phase honest thought stream unit tests
├── Dockerfile               # Pinned container with pre-cached models
├── docker-compose.yml       # Qdrant + Ingest + API with healthcheck
├── pyproject.toml           # Project dependencies & configurations
└── requirements.txt         # Pinned production requirements
```
