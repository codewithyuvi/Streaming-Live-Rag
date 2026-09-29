# Streaming Live RAG: Operational Runbook

This runbook provides complete operational instructions for running, testing, evaluating, demonstrating, and deploying the Streaming Live RAG system across all completed phases, including the live Side-by-Side VS Arena, real-time voice streaming, hardware GPU acceleration, and multi-format document management.

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

#### Option A: Direct Python (Recommended for Local Dev & GPU Acceleration)
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

*Windows One-Click Launcher: Double-click `start.bat` or run `.\start.ps1` from the repository root.*

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

## 2. Interactive Features & Workflows

### 2.1 Side-by-Side VS Arena (`/ws/dual_stream`)
The VS Arena pits **Streaming Live RAG** against **Naive Sequential RAG** simultaneously in real time.

1. Navigate to `http://localhost:8000` and click the **"VS Arena"** tab in the top navigation.
2. Select or enter a query (e.g., *"What is the capacity of the Pune venue for 30 attendees and what is the cancellation refund policy?"*).
3. Ensure **"Bypass Cache"** is checked (`ignore_cache=True`) for a completely fair, unbiased comparison.
4. Click **"Run VS Comparison"**.
5. Observe the live metrics updating concurrently:
   - **Streaming Live RAG (Left)**: Emits early thought badges at $t_1$, completes retrieval while utterance arrives, and achieves a **TTFT of ~300–450ms**.
   - **Naive Sequential RAG (Right)**: Sits idle until speech finishes, then performs monolithic search, achieving a **TTFT of ~2100–3400ms**.
   - Side-by-side latency meters (Time to First Token, Retrieval Time, Tokens/sec, Grounding Score).

### 2.2 Live Microphone Voice Input
1. In either the Live Chat or VS Arena tabs, click the **Microphone** icon.
2. Grant microphone permissions in your web browser when prompted.
3. Speak naturally into your microphone (e.g., *"Tell me about the memory segmentation policy and paging"*).
4. As you speak:
   - The audio visualizer animates with live voice amplitude waves.
   - Partial transcripts stream to the server every ~300ms.
   - Provisional retrieval fires speculatively at $t_1$.
5. Stop speaking. The system detects silence, closes the stream, and immediately streams the synthesized response with zero perceived waiting time.

### 2.3 Hardware GPU Acceleration
The system auto-detects NVIDIA GPUs (e.g. NVIDIA RTX 5060 Laptop GPU) and binds CUDA execution providers:
- Inspect status via the UI navbar badge (`⚡ GPU: NVIDIA GeForce RTX 5060 (8GB)`).
- Query the REST endpoint:
  ```bash
  curl http://localhost:8000/system/hardware
  ```
- Retrieval latency profile:
  - Dense embedding: **~3.0ms**
  - Cross-Encoder reranker: **~3.0ms**
  - Total hybrid search: **~10ms** (down from ~170ms on CPU).

---

## 3. Document Ingestion & Corpus Management

The corpus engine supports **PDF**, **TXT**, and **Markdown** documents with SHA-256 deduplication and strict baseline benchmark isolation:

### 3.1 Corpus Directory Layout
- **`data/dev_corpus/`**: Contains strictly the baseline benchmark corpus (`Doc_01_venue_booking.txt` and `Doc_02_travel_policy.txt`). On system boot, the vector collection `dev_corpus_dense` is automatically verified and reseeded to strictly these 4 chunks for 100% reproducible benchmark evaluations.
- **`data/sample_documents/`**: Holds reference PDFs (`Doc_03_Module_5_memory_management.pdf`, `Doc_03_Segmentation.pdf`, `Doc_03_Yuvraj_s_Resume.pdf`, `Doc_04_Module_4_Concurrency.pdf`).
- **`data/uploads/`**: Dynamic user uploads via UI or API are safely stored here with assigned canonical identifiers (e.g. `Doc_03`, `Doc_04`) without mutating the pristine benchmark files.

### 3.2 Ingesting and Resetting Corpus
1. **Reset to Clean Baseline (Doc 1 & Doc 2):**
   ```bash
   curl -X POST http://localhost:8000/corpus/reset
   ```
2. **Batch Ingestion via CLI:**
   ```bash
   python -m retrieval.ingest
   ```
3. **Upload Files via REST API or Dashboard:**
   ```bash
   curl -X POST http://localhost:8000/corpus/upload \
     -F "file=@my_notes.pdf"
   ```
4. **Inspect Corpus Summary:**
   ```bash
   curl http://localhost:8000/corpus
   # (also aliased at http://localhost:8000/documents/summary)
   ```
   Returns total chunks indexed, document filenames, and active canonical citations.

---

## 4. API Transports & Contracts

### 4.1 Live Dual-Stream WebSocket (`/ws/compare` or `/ws/dual_stream`)
Runs both pipelines simultaneously in parallel with complete session isolation:
- **Client -> Server:**
  ```json
  {"type": "chunk", "text": "What is the Pune venue capacity..."}
  {"type": "chunk", "text": "What is the Pune venue capacity and cancellation penalty?"}
  {"type": "end"}
  ```
  *(Optional: send `{"type": "cancel"}` to immediately terminate an active in-flight comparison turn).*
- **Server -> Client Events:**
  - `{"type": "session_init", "live_session": "cmp_voice_123_live", "norm_session": "cmp_voice_123_norm"}`
  - `{"lane": "live", "type": "thought", "phase": 1, ...}` (Emitted speculatively while speaking)
  - `{"lane": "normal", "type": "listening", "text": "User speaking... Classic sequential RAG is idle."}`
  - `{"type": "speech_ended", "duration_s": 2.8, "final_transcript": "..."}`
  - `{"lane": "live", "type": "done", "answer": "...", "wait_s": 0.42, "telemetry": {...}}`
  - `{"lane": "normal", "type": "done", "answer": "...", "wait_s": 2.85, "telemetry": {...}}`
  - `{"type": "summary", "speech_duration_s": 2.8, "live_wait_s": 0.42, "normal_wait_s": 2.85, "speedup_factor": 6.8}`

### 4.2 Live Single-Stream WebSocket (`/ws/stream`)
Full-duplex transport for real-time speech and 4-phase thought streaming:
- **Client -> Server:**
  - Start turn: `{"type": "start", "session_id": "sess_123", "live_mode": true}`
  - Speech chunk: `{"type": "chunk", "text": "What is the capacity..."}`
  - Speech finish: `{"type": "end"}`
  - Cancel turn: `{"type": "cancel"}`
- **Server -> Client Live Events:**
  - `stream_started`, `transcript`, `controller`, `thought` (Phases 0–4), `retrieval_started`, `answer`, `telemetry`.

### 4.3 HTTP Comparison Endpoint (`POST /turn/compare`)
Synchronously runs both pipelines over an utterance with simulated speech pacing and returns head-to-head metrics:
```bash
curl -X POST http://localhost:8000/turn/compare \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What is the venue booking capacity and cancellation penalty in Pune?",
    "words_per_chunk": 2,
    "ms_per_chunk": 200
  }'
```

### 4.4 HTTP Simulated Pacing Endpoint (`POST /turn`)
Replays an utterance with simulated speech pacing (2 words per chunk every 300ms):
```bash
curl -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess_demo_01",
    "turn_id": 1,
    "utterance": "What are the dimensions and capacity of the Pune venue?"
  }'
```

### 4.5 Session Management (`POST /session/reset`)
Wipes session conversation history and version counters:
```bash
# Reset specific session
curl -X POST "http://localhost:8000/session/reset?session_id=sess_demo_01"

# Or reset all active in-memory sessions
curl -X POST http://localhost:8000/session/reset
```

### 4.6 Hardware Diagnostics Endpoint (`GET /system/hardware`)
```bash
curl http://localhost:8000/system/hardware
```

---

## 5. Automated Testing Suite

The codebase includes comprehensive unit tests verifying the pipeline, session store, hybrid search, grounding validator, dual stream, GPU acceleration, and thought stream:

```bash
# Run all unit tests (26 passed)
pytest tests/ -v

# Run thought stream specific tests
pytest tests/test_thought_stream.py -v

# Run integration tests
pytest tests/test_pipeline.py -v
```

---

## 6. Master Evaluation Harness (Gates G1–G6)

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

## 7. Troubleshooting & FAQ

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `GPU indicator shows CPU mode` | Missing PyTorch CUDA libraries or non-NVIDIA host | The system safely falls back to CPU. To enable CUDA, ensure `torch` with CUDA 12 is installed (`pip install torch --index-url https://download.pytorch.org/whl/cu124`) and `onnxruntime-gpu` is installed. |
| `Microphone permission denied` | Browser blocked audio access | Click the lock/site settings icon in your browser address bar and set Microphone to "Allow". Note: on non-localhost origins, browsers require HTTPS (e.g. via Cloudflare tunnel). |
| `Remote Qdrant unreachable` | Docker Qdrant not running on `:6333` | The system automatically falls back to embedded storage in `data/qdrant_storage`. Zero manual setup needed. |
| `GROQ_API_KEY is not set` | Missing key in `.env` | Add key to `.env` or click "Provider Settings" in the Web UI to input your key via BYOK modal. |
| `ADMIN_TOKEN forbidden` | Non-localhost caller hitting admin routes | Set `ADMIN_TOKEN` in `.env` and pass `X-Admin-Token` header, or access directly via `localhost:8000`. |
