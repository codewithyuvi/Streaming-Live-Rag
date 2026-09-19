# ADR-2: Retrieval backend — Qdrant (self-hosted, one Docker service)

**Decision:** Qdrant, run as a single `docker-compose` service, is the vector database and the sparse/BM25 engine.

**Why:** The theme explicitly grades a ”Dense/Sparse Hybrid Scoring” pipeline stage, not just similarity search. Qdrant's Query API supports named dense + sparse vectors on the same collection with **server-side BM25** and single-call fusion (RRF or DBSF) via `prefetch` — meaning the ”hard” part of Component 3 (Corpus Retrieval & Fusion) is a well-tested, already-built primitive rather than code you write and debug under time pressure. This collapses what would otherwise be two systems (a vector index plus a separate BM25 library) into one Docker container, which directly serves both G1 reproducibility and the parsimony rule. General 2026 vector-DB guidance for hackathons often points to FAISS as the default ”just get something working” choice — but that advice assumes dense-only retrieval is enough; here, hybrid fusion quality is itself graded, so the calculus is different.

**Alternatives considered:** 
- FAISS + a standalone BM25 library (`bm25s`, actively maintained and dramatically faster than the older `rank_bm25`) — genuinely fine and the recommended **fallback** if Qdrant/Docker networking becomes a blocker; you lose the one-call fusion but gain zero server dependencies. 
- `pgvector` — solid in general, but adds Postgres operational overhead here for no offsetting benefit, since you don't need relational joins.

**Known sharp edges (from Qdrant's own guidance):** BM25's `avg_len` parameter is *not* computed automatically — calibrate it to your actual chunk length, or short chunks get systematically over/under-scored. Sparse learned alternatives to BM25 (BM42, SPLADE) exist but are English-only, need domain fine-tuning, and in BM42's case are explicitly unmaintained — don't reach for them under time pressure.

**Revisit if:** corpus turns out to be enormous (tens of millions of chunks) or multi-tenant isolation becomes a real requirement — neither is expected here.
