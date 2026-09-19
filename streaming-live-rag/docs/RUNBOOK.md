# Streaming Live RAG: Project Runbook

This document explains how to run the project scripts and servers for each phase of development.

## Prerequisites
Before running anything, ensure your environment is set up:
```bash
# 1. Start Qdrant in the background
docker compose up -d

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Ensure your `.env` file has the Gemini API Key
```

---

## Phase 0: Proof of Concepts (Gate 1)
These scripts were used to validate the underlying database and LLM infrastructure.

**Run Qdrant POC:**
```bash
python scripts/poc_a_qdrant.py
```

**Run Dual-Provider LLM POC:**
```bash
python scripts/poc_b_llm.py
```

---

## Phase 1 & 2: Baseline Hybrid Retrieval (Current Phase)
This phase implements the FastAPI server, Dense+Sparse embeddings, and MiniLM cross-encoder reranking.

**Step 1: Ingest the Corpus**
You must run the ingester to read the raw `.txt` files in `data/dev_corpus`, generate BGE dense embeddings and BM25 sparse embeddings, and push them to Qdrant.
```bash
python retrieval/ingest.py
```

**Step 2: Start the FastAPI Server**
Start the main application server. It will hot-reload automatically if you edit `api/main.py`.
```bash
uvicorn api.main:app --reload
```

**Step 3: Test the Endpoint**
In a separate terminal, test the API using a `curl` request:
```bash
curl -X POST http://127.0.0.1:8000/turn \
-H "Content-Type: application/json" \
-d '{"session_id": "test_03", "turn_id": 1, "utterance": "What are the dimensions of the venue?"}'
```

---

## Phase 3: Multi-Turn State (Upcoming)
*(Commands will be added here once Phase 3 is completed)*

## Phase 4: Async Decomposition (Upcoming)
*(Commands will be added here once Phase 4 is completed)*

## Phase 5: Streaming & Interruption (Upcoming)
*(Commands will be added here once Phase 5 is completed)*
