# ⚡ Streaming Live RAG — Theme 4
**Samsung PRISM GenAI Hackathon 2026–27**  
*A low-latency, session-aware retrieval-augmented generation pipeline with real-time speech controller, multi-intent decomposition, and deterministic grounding verification.*

---

## 🚀 3-Command Quickstart (Gate G1 Reproducibility)

```bash
# 1. Configure environment keys
cp .env.example .env
# (Add your GROQ_API_KEY and GEMINI_API_KEY in .env)

# 2. Boot system (Qdrant + One-Shot Ingestion + FastAPI with Healthcheck)
docker compose up --build

# 3. Run full benchmark evaluation (Gates G1–G6)
./run_eval.sh      # On Linux / macOS
.\run_eval.bat     # On Windows
```

Once running, access the **Interactive Live Demo UI** at:  
👉 **`http://localhost:8000`** (or `http://localhost:8000/demo`)

---

## 📊 Benchmark Scorecard (G1 + G6 Passed, G2–G5 Require Live API Keys)

| Gate | Name | Target | Measured | Result |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | **Reproducibility & Packaging** | Single-command clean boot, 100% | **100.0%** | 🟢 **PASSED** |
| **G2** | **Early Retrieval Trigger** | Target: eligible queries, 0% false triggers | Skipped — needs `GROQ_API_KEY` | ⚪ **SKIPPED** |
| **G3** | **Multi-Intent Decomposition** | Target: compound queries | Skipped — needs `GROQ_API_KEY` | ⚪ **SKIPPED** |
| **G4** | **Grounding Validation** | Target: support, 0 fabricated IDs | Skipped — needs `GEMINI_API_KEY` | ⚪ **SKIPPED** |
| **G5** | **Session Refinement** | Target: correct refinement & suppression | Skipped — needs `GROQ_API_KEY` | ⚪ **SKIPPED** |
| **G6** | **Telemetry Observability** | 100% trace coverage & persistence | **100.0%** | 🟢 **PASSED** |

Detailed scorecard results are saved to: `eval/results/scorecard.json`.

---

## 🏗️ Architecture & Pipeline Flow

```
User Audio / ASR Stream (300ms chunks)
                │
                ▼
  ┌───────────────────────────────┐
  │   Streaming Controller (G2)   │
  │  - Stability stop-word guard  │
  │  - Early provisional search   │
  │  - Chit-chat early return     │
  └───────────────┬───────────────┘
                  │
        Utterance Clock t_end
                  │
                  ▼
  ┌───────────────────────────────┐
  │  Refinement & Intent Routing  │
  │  - NEW_TOPIC / LATE_DETAIL    │
  │  - PRESENTATION_ONLY          │
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │ Multi-Intent Decomposer (G3)  │
  │  - 1..4 orthogonal sub-queries│
  │  - asyncio.gather retrieval   │
  │  - Quota merge (min 2 per sq) │
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │  Grounded Synthesis & G4      │
  │  - Gemini Flash synthesis     │
  │  - Claim-level validator      │
  │  - Regenerate / abstain logic │
  └───────────────┬───────────────┘
                  │
                  ▼
        Streamed Answer + Citations
        + Telemetry Event to JSONL (G6)
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

---

## 📁 Repository Structure

```
streaming-live-rag/
├── api/
│   └── main.py              # FastAPI live pipeline & /turn endpoint
├── controller/
│   ├── decide.py            # Stream controller (Wait/Retrieve/Suppress)
│   ├── decompose.py         # Multi-intent query decomposer
│   ├── heuristics.py        # Stability heuristics & stop-word guards
│   └── refinement.py       # Refinement classifier (New/Late/Presentation)
├── retrieval/
│   ├── grounding.py         # Claim-level grounding validator (ADR-5)
│   ├── hybrid_search.py     # FastEmbed dense + BM25 RRF & reranker
│   ├── ingest.py            # Corpus parser & Qdrant ingestion (IDF BM25)
│   └── merge.py             # Sub-intent quota merger
├── session/
│   └── store.py             # Ephemeral in-memory session state
├── static/
│   └── index.html           # Interactive demo dashboard UI
├── telemetry/
│   ├── schema.py            # TelemetryEvent Pydantic schema
│   └── sink.py              # JSONL persistence sink
├── eval/
│   ├── dataset_loader.py    # Robust dataset loader
│   ├── labeled_set.yaml     # 64 benchmark queries & ground truth
│   ├── report.py            # Scorecard formatter
│   ├── run_eval.py          # Master evaluation runner (G1-G6)
│   └── gates/               # Gate verification scripts (g1 to g6)
├── Dockerfile               # Pinned container with pre-cached models
├── docker-compose.yml       # Qdrant + Ingest + API with healthcheck
├── pyproject.toml           # Project dependencies
└── README.md
```
