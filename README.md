# ⚡ Streaming Live RAG — Theme 4
**Samsung PRISM GenAI Hackathon 2026–27**  
*A low-latency, session-aware retrieval-augmented generation pipeline with real-time speech controller, 4-phase honest thought stream, multi-intent decomposition, deterministic grounding verification, GPU hardware acceleration, and a live side-by-side VS Arena.*

---

## 🌟 Overview & Key Innovations

Streaming Live RAG addresses the fundamental challenge in real-time conversational voice and text systems: **latency vs. factual accuracy**. Conventional RAG systems wait until the user finishes speaking, run a monolithic vector search, and then prompt an LLM.

Our pipeline changes this paradigm:
1. **🥊 Live Side-by-Side VS Arena (`/ws/compare` or `/ws/dual_stream`, `POST /turn/compare`)**:
   - Executes **Streaming Live RAG** and **Naive Sequential RAG** simultaneously in real time.
   - Independent sessions (`sess_left` and `sess_right`) run concurrently via `asyncio.gather`.
   - **Cache-Bypass Option (`ignore_cache=True`)**: Enforces fair benchmarking without either pipeline benefiting from warm vector cache hits.
   - Real-time side-by-side comparison gauges: Time-to-First-Token (TTFT), retrieval latency, tokens per second, and grounding scores.
2. **🎙️ Synchronized Live Voice & Audio Streaming**:
   - Direct browser microphone capture via the **Web Speech API & MediaRecorder** with real-time waveform visualization.
   - Streams speech transcript chunks simultaneously to both pipelines.
   - Directly experience the massive real-world TTFT advantage: provisional retrieval launches *while you are still speaking*, so answers begin streaming the instant you finish.
   - **Continuous Speculative Retrieval**: Dynamically detects multiple intent clauses mid-stream, executing parallel speculative searches in the background during active speech.
3. **🚀 Hardware GPU Acceleration (NVIDIA CUDA & DirectML)**:
   - Auto-detects NVIDIA GPUs (e.g. RTX 5060 Laptop GPU) and DirectML devices.
   - FastEmbed dense embeddings (`BAAI/bge-small-en-v1.5`) and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) execute in **~3.0ms** (down from ~50–120ms on CPU).
   - Windows PyTorch CUDA 12 dynamic library loading (`cublas64_12.dll`, `cudart64_12.dll`, `cudnn`) with graceful silent fallback to CPU.
   - Dedicated hardware diagnostics endpoint (`GET /system/hardware`) and live UI navbar badge.
4. **🧠 4-Phase Honest Thought Stream**:
   - Emits structured, timestamped cognitive events in real time across WebSocket streaming (`/ws/stream`) and pacing-simulated HTTP turns (`/turn`):
     - 🔍 `intent_detected`: Real-time intent classification mid-utterance (~300ms chunks).
     - ⚡ `provisional_search`: Early speculative dense + sparse retrieval at $t_1$.
     - 🧩 `decomposition_planned`: Multi-intent query breakdown on compound utterances.
     - 🛡️ `synthesis_ready`: Grounded answer synthesis with citation validation and confidence scoring.
5. **📚 Multi-Format Document Ingestion & Deduplication**:
   - Ingests **PDF**, **TXT**, and **Markdown** documents with SHA-256 content deduplication and stable Doc IDs (`Doc_01` to `Doc_06`).
   - Vector database scaled to 173+ chunks indexed across technical systems, policies, and resumes.
6. **🛡️ Deterministic Grounding (Gate G4) & Ephemeral Session Memory (Gate G5)**:
   - Deterministic 44-line regex validator eliminates fabricated citations and verifies claim support.
   - Ephemeral session memory with clean version lineage (`NEW_TOPIC` resets to $v=1$, `LATE_DETAIL` increments to $v=2$ with citation unioning, `PRESENTATION_ONLY` bypasses retrieval).
7. **💾 Zero-Dependency Embedded Fallback**:
   - Automatically falls back from remote Qdrant to local on-disk storage (`data/qdrant_storage`), allowing the entire pipeline to run natively without requiring Docker.

---

## 🚀 Quickstart

You can run Streaming Live RAG via **Direct Python** (recommended for local development and GPU support) or **Docker Compose**.

### Option 1: Direct Python (Fastest, Full GPU Acceleration)

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

*Windows Quick Launch: Simply run `start.bat` or `.\start.ps1` from the root directory.*

Once running, open:
- 🌐 **Interactive Dashboard & VS Arena:** [http://localhost:8000](http://localhost:8000)
- 🩺 **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- 💻 **Hardware Diagnostics:** [http://localhost:8000/system/hardware](http://localhost:8000/system/hardware)
- 📑 **Swagger API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Docker Compose (Single-Command Production Boot)

```bash
cd streaming-live-rag
cp .env.example .env
docker compose up --build -d
```

### Option 3: Expose via Cloudflare Tunnel

To share a live interactive demo link with evaluators or mobile devices:
```bash
cloudflared tunnel --url http://localhost:8000
```
This generates a public `.trycloudflare.com` URL routing directly to your local instance.

---

## 🧪 Testing & Evaluation

### Run Unit Tests
The test suite verifies the pipeline, hybrid search, grounding validator, thought stream, dual stream, and GPU hardware detection:
```bash
cd streaming-live-rag
pytest tests/ -v
# 26 passed in ~3s
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

## 🥊 Head-to-Head: Streaming Live RAG vs. Naive Sequential RAG

| Dimension | Streaming Live RAG | Naive Sequential RAG | Advantage |
| :--- | :--- | :--- | :--- |
| **Retrieval Triggering** | Early speculative at $t_1$ while user is still speaking | Waits until speech completes ($t_{end}$) | **~1.5s to 2.5s head start** |
| **Time to First Token (TTFT)** | **~300 – 450 ms** | **~2100 – 3400 ms** | **5x to 7x faster** |
| **User Experience** | Real-time cognitive narration & instantaneous tokens | Prolonged silence & waiting spinner | **Fluid conversational flow** |
| **Compound Inquiries** | Multi-intent decomposition with parallel quota retrieval | Monolithic retrieval prone to topical drift | **Complete multi-topic answers** |
| **Multi-Turn Sessions** | Context-aware version progression ($1 \to 1 \to 2$) & citation unioning | Stateless or brittle context-window padding | **Auditable version history** |
| **Inference Hardware** | **CUDA GPU Accelerated** (Embeddings & Reranking in ~3ms) | CPU baseline (~50–120ms) | **~20x faster search inference** |

---

## 🔌 API Transports & Contracts

### 1. Dual-Stream Live Arena: `WebSocket /ws/compare` or `/ws/dual_stream` (and `POST /turn/compare`)
Runs Streaming Live RAG and Naive RAG side-by-side simultaneously.
- **Client sends:**
  - `{"type": "chunk", "text": "What is the Pune venue capacity..."}`
  - `{"type": "chunk", "text": "What is the Pune venue capacity and cancellation penalty?"}`
  - `{"type": "end"}`
  *(Optional: send `{"type": "cancel"}` to immediately abort turn).*
- **Server emits for both streams concurrently:**
  - `{"type": "session_init", "live_session": "...", "norm_session": "..."}`
  - `{"lane": "live", "type": "thought", "phase": 1, ...}` (Emitted speculatively while user speaks)
  - `{"lane": "normal", "type": "listening", "text": "User speaking... Classic sequential RAG is idle."}`
  - `{"lane": "live", "type": "done", "answer": "...", "wait_s": 0.35, "telemetry": {...}}`
  - `{"lane": "normal", "type": "done", "answer": "...", "wait_s": 2.45, "telemetry": {...}}`
  - `{"type": "summary", "speech_duration_s": 2.8, "live_wait_s": 0.35, "normal_wait_s": 2.45, "speedup_factor": 7.0}`

### 2. Single-Stream Live Engine: `WebSocket /ws/stream`
Full-duplex transport for real-time speech and thought streaming:
- **Client sends:**
  - `{"type": "start", "session_id": "sess_123"}`
  - `{"type": "chunk", "text": "cumulative speech..."}`
  - `{"type": "end"}`
- **Server emits:**
  - `{"type": "transcript", ...}`
  - `{"type": "controller", ...}`
  - `{"type": "thought", "stage": "...", ...}`
  - `{"type": "retrieval", ...}`
  - `{"type": "answer", ...}`
  - `{"type": "telemetry", ...}`

### 3. Hardware Diagnostics: `GET /system/hardware`
Returns real-time GPU and ONNX execution provider status:
```json
{
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "vram_total_gb": 8.0,
  "directml_available": false,
  "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "active_provider": "CUDAExecutionProvider"
}
```

### 4. Dynamic BYOK Configuration: `/config/llm`
Update LLM providers, models, or API keys at runtime without restarting the server:
- `GET /config/llm`: Inspect active providers and models (write-only keys are masked).
- `POST /config/llm`: Update credentials/models.
- `POST /config/llm/test`: Ping providers to measure live latency.

---

## 📁 Repository Structure

```
Streaming-Live-Rag/
├── README.md                      # Primary repository landing page & documentation
├── start.bat                      # Windows one-click batch launcher
├── start.ps1                      # Windows PowerShell launcher
└── streaming-live-rag/
    ├── api/
    │   └── main.py                # FastAPI routes, /ws/stream, /ws/compare, /ws/dual_stream, /system/hardware
    ├── controller/
    │   ├── decide.py              # Two-stage controller (Wait / Retrieve / Suppress)
    │   ├── decompose.py           # Multi-intent query decomposer
    │   ├── heuristics.py          # Trailing stop-word & stability guards
    │   └── refinement.py          # Session turn classifier (New / Late / Presentation)
    ├── data/
    │   ├── dev_corpus/            # Baseline benchmark documents (Doc 1 & Doc 2)
    │   ├── sample_documents/      # Reference technical PDFs (OS, concurrency, resumes)
    │   ├── uploads/               # Dynamic user upload storage
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
    │   ├── hybrid_search.py       # Dense + BM25 sparse RRF fusion & GPU reranker
    │   ├── ingest.py              # Multi-format parser (PDF, TXT) & SHA-256 deduplication
    │   ├── merge.py               # Multi-intent quota merger
    │   └── parsers.py             # Document format parsers
    ├── session/
    │   └── store.py               # In-memory session state, cache bypass, & version lineage
    ├── static/
    │   └── index.html             # Interactive demo dashboard with VS Arena & Live Voice
    ├── streaming/
    │   ├── engine.py              # Unified turn engine, dual stream, & thought streamer
    │   └── live_stream.py         # Async live queue & pacing utterance playback
    ├── telemetry/
    │   ├── schema.py              # Strict Pydantic TelemetryEvent schema
    │   └── sink.py                # JSONL telemetry persistence sink
    ├── tests/
    │   ├── test_pipeline.py       # End-to-end integration test suite (26 tests)
    │   └── test_thought_stream.py # 4-phase honest thought stream unit tests
    ├── Dockerfile                 # Pinned container with cached FastEmbed models
    ├── docker-compose.yml         # Qdrant + FastAPI multi-container stack
    ├── pyproject.toml             # Project dependencies & tool configurations
    └── requirements.txt           # Pinned production requirements
```

---

## 🔒 Security & Guardrails

- **Zero Data Leakage:** Sessions are strictly ephemeral, keyed by isolated `session_id`, and automatically cleaned up.
- **Admin Access Control:** Constant-time `hmac.compare_digest` verification for administrative operations (`/config/llm`, `/corpus/clear`, `/upload`). Loops back strictly to `localhost` and local bridge subnets when `ADMIN_TOKEN` is unset.
- **Rate Limiting & DoS Protection:** Dual-bucket rate limiter enforcing 60 requests/minute per (IP, session) and 300 requests/minute per egress IP with automated TTL sweeps.
- **Corpus Grounding Isolation:** All answers are derived strictly from the indexed local corpus; hallucinations and fabricated citation tags are trapped by the deterministic grounding engine.
