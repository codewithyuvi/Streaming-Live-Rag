# ADR-3: Embeddings + reranker — FastEmbed (ONNX), not a PyTorch/sentence-transformers stack

**Decision:** `qdrant-client[fastembed]` for dense embeddings (`BAAI/bge-small-en-v1.5`, FastEmbed's own default), native Qdrant BM25 for sparse, and FastEmbed's `TextCrossEncoder` (`ms-marco-MiniLM-L-6-v2`, ONNX) for reranking.

**Why:** FastEmbed is built and maintained by the Qdrant team specifically to pair with Qdrant, runs on the ONNX runtime (no multi-GB PyTorch/CUDA install), and covers dense, sparse, *and* cross-encoder reranking in one lightweight package — which matters enormously for a ”launches with one command on a clean machine” reproducibility gate. `bge-small-en-v1.5` is a well-established, CPU-fast, MTEB-competitive 384-dim model; `ms-marco-MiniLM-L-6-v2` is the widely-recommended default fast reranker (roughly 100–300ms for 50 candidates on CPU) with the best speed/quality balance for a latency-sensitive theme.

**Caution, and your first honest ablation opportunity:** rerankers are not automatically a win — at least one 2026 study found off-the-shelf cross-encoders hurting nDCG on out-of-domain corpora while adding real latency. Don't assume reranking helps; measure it on your own eval set (Phase 2) before committing, exactly as you're already required to do for the hybrid-vs-dense ablation.

**Alternatives considered:** `BAAI/bge-reranker-v2-m3` (stronger, multilingual, heavier — a good Day-8 upgrade *if* your eval set shows it beats MiniLM and you have GPU headroom) — keep it in your back pocket, not your Day-1 default.

**Revisit if:** the corpus turns out to be multilingual, or a GPU is confirmed available and Day-2/3 benchmarking shows a real quality gap worth the latency.
