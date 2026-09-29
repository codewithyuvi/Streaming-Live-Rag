# ⚡ Streaming Live RAG — Theme 4
**Samsung PRISM GenAI Hackathon 2026–27**  
*A low-latency, session-aware retrieval-augmented generation pipeline with real-time speech controller, 4-phase honest thought stream, multi-intent decomposition, deterministic grounding verification, GPU hardware acceleration, and a live side-by-side VS Arena.*

---

## 🌟 Key Highlights & Innovations

1. **🥊 Live Side-by-Side VS Arena (`/ws/compare` or `/ws/dual_stream`, `POST /turn/compare`)**:
   - Compares **Streaming Live RAG** against **Naive Sequential RAG** running simultaneously in real time.
   - **True Concurrent Execution**: Both pipelines run in parallel via `asyncio.gather` on independent sessions (`sess_left` and `sess_right`).
   - **Cache-Bypass Guarantee (`ignore_cache=True`)**: Completely eliminates warm cache advantages to ensure a 100% fair head-to-head comparison.
   - **Live Latency Telemetry**: Real-time side-by-side Time-to-First-Token (TTFT), retrieval duration, generation speed (tokens/sec), and deterministic grounding scores.

2. **🎙️ Synchronized Live Voice & Audio Streaming**:
   - Built-in browser **Web Speech API & Audio Streaming** captures real-time microphone input with real-time waveform visualization.
   - Streams audio transcript chunks incrementally to both pipelines simultaneously.
   - Experience the true power of **Early Provisional Retrieval (Gate G2)**: while you are still speaking, the streaming engine detects stable semantic prefixes and fetches candidate chunks at $t_1$, achieving instantaneous answers as soon as speech finishes.
   - **Continuous Speculative Retrieval**: Intelligently extracts multiple intent clauses mid-stream, firing parallel searches in the background while speech is in progress.

3. **🚀 Hardware GPU Acceleration (NVIDIA CUDA & DirectML)**:
   - Auto-detects NVIDIA GPUs (e.g., NVIDIA GeForce RTX 5060 Laptop GPU) and DirectML devices.
   - Accelerates FastEmbed dense embeddings (`BAAI/bge-small-en-v1.5`) and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) down to **~3.0ms** (a ~20x speedup over CPU).
   - Dynamic Windows PyTorch CUDA 12 library resolution (`cublas64_12.dll`, `cudart64_12.dll`, `cudnn`) with graceful silent fallback to CPU.
   - Live hardware status badge in the UI and dedicated diagnostics endpoint at `GET /system/hardware`.

4. **🧠 4-Phase Honest Thought Stream**:
   - Emits structured, timestamped cognitive events in real time without artificial delays:
     - 🔍 `intent_detected`: Speculative user intent classification mid-utterance (~300ms chunks).
     - ⚡ `provisional_search`: Early dense + sparse hybrid retrieval launched at $t_1$ before utterance completion.
     - 🧩 `decomposition_planned`: Multi-intent query breakdown on compound utterances.
     - 🛡️ `synthesis_ready`: Grounded answer synthesis with citation validation and confidence scoring.

5. **📚 Multi-Format Document Ingestion & Deduplication**:
   - Supports **PDF**, **TXT**, and **Markdown** documents.
   - Robust content-hash (SHA-256) deduplication prevents redundant embeddings during updates while preserving stable document identifiers (`Doc_01` through `Doc_06`).
   - Expanded active vector database with 173+ chunks indexed across technical documents, policies, and resumes.

6. **🛡️ Deterministic Grounding & Ephemeral Session Memory**:
   - Strict 44-line deterministic validator eliminates fabricated citations and verifies claim support.
   - Ephemeral session memory with version lineage (`NEW_TOPIC` resets to $v=1$, `LATE_DETAIL` increments to $v=2$ with citation unioning, `PRESENTATION_ONLY` bypasses retrieval).

---

## 🚀 Quickstart

### Option 1: Direct Python (Recommended, Fast & GPU-Enabled)

```bash
# 1. Configure environment keys
cp .env.example .env
# Set GROQ_API_KEY and GEMINI_API_KEY in .env (or configure at runtime via UI settings)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Boot server (Auto-seeds corpus on startup into embedded local Qdrant storage)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

*On Windows, you can also simply double-click `start.bat` or run `.\start.ps1` from the repository root.*

### Option 2: Docker Compose (Gate G1 Containerized Boot)

```bash
# 1. Configure environment keys
cp .env.example .env

# 2. Boot container stack (Qdrant + FastEmbed + FastAPI)
docker compose up --build -d

# 3. Run full benchmark evaluation (Gates G1–G6)
./run_eval.sh      # On Linux / macOS
.\run_eval.bat     # On Windows
```

Once running, access the application:
- 🌐 **Interactive Live Dashboard & Arena:** [http://localhost:8000](http://localhost:8000)
- 🩺 **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- 💻 **Hardware Diagnostics:** [http://localhost:8000/system/hardware](http://localhost:8000/system/hardware)
- 📑 **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Expose via Cloudflare Tunnel
```bash
cloudflared tunnel --url http://localhost:8000
```

---

## 📊 Benchmark Scorecard (Gates G1–G6)

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

## 🥊 Streaming RAG vs. Naive Sequential RAG

| Metric / Dimension | Streaming Live RAG (Our System) | Naive Sequential RAG |
| :--- | :--- | :--- |
| **Retrieval Trigger** | **Early Speculative ($t_1$)**: Fires mid-utterance on stable prefix | **Monolithic ($t_{end}$)**: Waits for 100% of speech to finish |
| **Time-to-First-Token (TTFT)** | **~300 – 450 ms** (Instantaneous start upon pause) | **~2100 – 3400 ms** (Long awkward silence) |
| **Cognitive Feedback** | **4-Phase Honest Thought Stream** in real time | Opaque spinner / silent delay |
| **Query Complexity** | Decomposes compound queries into 1..4 parallel sub-queries | Monolithic search with potential semantic drift |
| **Multi-Turn Refinement** | Versioned session memory ($1 \to 1 \to 2$) & citation unioning | Stateless or context-overflowing full reprompt |
| **Hardware Acceleration** | **CUDA GPU Accelerated** (Embeddings & Reranking in ~3ms) | CPU baseline (~50–120ms) |
| **Cache Isolation** | Supports `ignore_cache=True` for unbiased testing | Prone to cache-warm bias |

---

## 🏗️ Architecture & Pipeline Flow

```
User Voice / Text Stream (300ms chunks)
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
    │    GPU Hybrid Retrieval & Fusion       │
    │  - FastEmbed BGE Dense (CUDA ~3ms)     │
    │  - Qdrant Sparse BM25 (IDF modifier)   │
    │  - Reciprocal Rank Fusion (RRF)        │
    │  - MiniLM Cross-Encoder Rerank (~3ms)  │
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │   Grounded Synthesis & Validator (G4)  │
    │  - Gemini Flash synthesis              │
    │  - Deterministic 44-line validator     │
    │  - Session state update (G5, v=1 -> 2) │
    │  - Emits: synthesis_ready + Answer     │
    └────────────────────────────────────────┘
```

---

## 🧪 Automated Testing

Run the full integration test suite:
```bash
pytest tests/ -v
```

All 26 unit tests verify:
- Live streaming and pacing simulation
- 4-phase honest thought narration progression
- Early provisional triggering and delta retrieval merging
- Citation extraction and zero fabricated IDs
- Ephemeral session versioning and citation unioning
- Side-by-side dual stream execution and cache bypass
- GPU acceleration detection and silent CPU fallback
- PDF and text document ingestion deduplication

---

## 📁 Repository Structure

```
streaming-live-rag/
├── api/
│   └── main.py              # FastAPI routes, /ws/stream, /ws/compare, /ws/dual_stream, /system/hardware
├── controller/
│   ├── decide.py            # Stream controller (Wait/Retrieve/Suppress)
│   ├── decompose.py         # Multi-intent query decomposer
│   ├── heuristics.py        # Stability heuristics & stop-word guards
│   └── refinement.py       # Refinement classifier (New/Late/Presentation)
├── data/
│   ├── dev_corpus/          # Baseline benchmark documents (Doc 1 & Doc 2)
│   ├── sample_documents/    # Reference technical PDFs (OS, concurrency, resumes)
│   ├── uploads/             # Dynamic user upload storage
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
│   ├── hybrid_search.py     # FastEmbed dense + BM25 RRF & GPU reranker
│   ├── ingest.py            # Multi-format parser (PDF, TXT) & SHA-256 deduplication
│   ├── merge.py             # Sub-intent quota merger
│   └── parsers.py           # Document format parsers
├── session/
│   └── store.py             # Ephemeral in-memory session state & cache bypass
├── static/
│   └── index.html           # Interactive demo dashboard UI with VS Arena & Voice
├── streaming/
│   ├── engine.py            # Dual-stream engine, thought streamer, and naive baseline
│   └── live_stream.py       # Async live stream queue & pacing utterance playback
├── telemetry/
│   ├── schema.py            # TelemetryEvent Pydantic schema
│   └── sink.py              # JSONL persistence sink
├── tests/
│   ├── test_pipeline.py     # End-to-end integration tests (26 passed)
│   └── test_thought_stream.py# 4-phase honest thought stream unit tests
├── Dockerfile               # Pinned container with pre-cached models
├── docker-compose.yml       # Qdrant + Ingest + API with healthcheck
├── pyproject.toml           # Project dependencies & configurations
└── requirements.txt         # Pinned production requirements
```
