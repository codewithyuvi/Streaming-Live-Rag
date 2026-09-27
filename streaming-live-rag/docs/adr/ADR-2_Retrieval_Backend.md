# ADR-2: Retrieval backend — Qdrant (Self-Hosted Docker with Embedded Local Storage Fallback)

**Status:** Accepted & Enhanced (September 2026)

**Decision:** Qdrant is the primary vector database and sparse BM25 engine, configured to support both containerized deployments (`docker compose`) and zero-Docker environments via automatic embedded local storage (`data/qdrant_storage`).

---

### Why:
The theme explicitly grades a **Dense/Sparse Hybrid Scoring** pipeline stage, not just similarity search. Qdrant's Query API supports named dense + sparse vectors on the same collection with **server-side BM25** and single-call fusion (Reciprocal Rank Fusion — RRF) via `prefetch`. This collapses what would otherwise be two systems (a vector index plus a separate BM25 library) into one unified system, directly serving Gate G1 reproducibility and the architectural parsimony rule.

---

### Architectural Enhancement: Dual-Mode Operation
During testing on varied host operating systems (e.g., Windows without WSL2 or when Docker Desktop is halted), relying solely on a network socket to `localhost:6333` presented an operational single-point-of-failure.

To resolve this without rewriting our hybrid queries, we implemented **transparent auto-fallback** in `retrieval/hybrid_search.py`:
1. **Primary Mode:** Attempts connection to remote Qdrant at `QDRANT_URL` (default: `http://localhost:6333`).
2. **Embedded Fallback Mode:** If remote Qdrant is unreachable, `get_qdrant_client()` automatically instantiates an embedded Qdrant instance storing vectors locally on disk at `data/qdrant_storage`.
3. **Parity:** Both modes expose the identical `QdrantClient` Python API, preserving dense vectors (BGE-Small 384d), sparse vectors (BM25 with `Modifier.IDF`), and RRF fusion without any code branches in retrieval logic.

---

### Alternatives Considered:
- **FAISS + standalone BM25 (`bm25s`):** Viable fallback, but lacks single-call RRF fusion and requires managing two separate vector/index serialization formats.
- **pgvector:** Introduces PostgreSQL database administration overhead with no offsetting advantage for a flat-file corpus.

---

### Known Sharp Edges & Mitigations:
- **BM25 IDF Weighting:** Standard BM25 requires corpus-wide document frequency. FastEmbed's `Qdrant/bm25` model paired with Qdrant's `Modifier.IDF` automatically calculates inverse document frequencies across ingested chunks.
- **Embedded Concurrency:** On-disk storage acquires a process lock. Handled safely in Uvicorn using asynchronous thread pools (`asyncio.to_thread`) and application startup lifecycle auto-seeding.
