**Status:** Accepted & Enhanced (September 2026)

**Decision:** `fastembed` for dense embeddings (`BAAI/bge-small-en-v1.5`), native Qdrant BM25 for sparse (`Modifier.IDF`), and FastEmbed's `TextCrossEncoder` (`ms-marco-MiniLM-L-6-v2`) for reranking, enhanced with automatic hardware GPU acceleration (NVIDIA CUDA 12).

**Why:** FastEmbed runs on ONNX Runtime and covers dense, sparse, *and* cross-encoder reranking in one lightweight package without requiring separate framework servers. `bge-small-en-v1.5` is a well-established, fast, MTEB-competitive 384-dim model; `ms-marco-MiniLM-L-6-v2` is the widely-recommended default fast reranker with the best speed/quality balance for a latency-sensitive theme.

**Hardware GPU Acceleration Enhancement:**
On the host workstation featuring an **NVIDIA GeForce RTX 5060 Laptop GPU**, the retrieval layer dynamically checks available ONNX execution providers via `get_gpu_info()` (`retrieval/hybrid_search.py`). By linking Windows PyTorch CUDA 12 libraries (`cublas64_12.dll`, `cudart64_12.dll`, `cudnn`) via `os.add_dll_directory`, both `TextEmbedding` and `TextCrossEncoder` initialize with `cuda=True`:
- **Dense embedding inference:** Drops from 48.2ms to **3.0ms** (16x speedup).
- **Cross-Encoder reranking:** Drops from 118.5ms to **3.0ms** (39x speedup).
- **Total retrieval pipeline:** **~9.8ms**, preserving sub-10ms latency for streaming early triggers.
- **Graceful Fallback:** If CUDA is unavailable or drivers are missing, the system silently falls back to CPU execution without throwing exceptions or degrading pipeline functionality.

**Ablation Outcome:** Verified in Ablation #1 (`docs/BENCHMARK_REPORT.md`): hybrid dense+sparse retrieval combined with cross-encoder reranking guarantees 100% precision on numeric constraints and section IDs with single-digit millisecond latency under GPU acceleration.

