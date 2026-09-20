# Ablation: Hybrid Retrieval vs Dense-Only Retrieval

As part of the Phase 2 (Gate 3) requirements, we conducted an ablation study to formally evaluate the necessity and impact of upgrading from a Dense-Only search architecture to a Hybrid Search architecture with Cross-Encoder reranking.

## Benchmark Configuration
- **Dataset**: `eval/labeled_set.yaml` (14 evaluatable queries)
- **Dense-Only Pipeline**: BGE-Small-EN dense vectors queried directly against Qdrant.
- **Hybrid+Rerank Pipeline**: BGE-Small-EN (Dense) + BM25 (Sparse) queried via Qdrant Reciprocal Rank Fusion (RRF), followed by reranking the top candidates using a `MiniLM-L-6-v2` cross-encoder.

## Results

| Metric | Dense-Only | Hybrid + Rerank |
|---|---|---|
| **Recall@3** | 100.00% | 100.00% |
| **Average Latency** | 18.35ms | 82.08ms |

## Analysis
The benchmark reveals several key insights:

1. **Recall Parity on Simple Queries**: Both approaches achieved a 100% hit rate for the top 3 chunks. The dataset currently consists largely of straightforward keyword-heavy queries where basic semantic similarity is already highly effective. 
2. **Latency Tradeoff**: The Hybrid pipeline introduces approximately ~64ms of overhead. This overhead stems from two factors:
   - The parallel execution and fusion of sparse vectors within Qdrant.
   - The deep neural network inference required by the cross-encoder to rescore the candidates.

## Conclusion
While Dense-Only is faster, we have opted to **retain the Hybrid + Rerank architecture**. As we move into Phase 4 (Multi-Intent Decomposition), queries will become increasingly fragmented and complex. The lexical exact-match capabilities of BM25 (Sparse) and the strict precision of the Cross-Encoder will be critical for maintaining high recall when the context becomes highly ambiguous. The ~80ms latency is well within our 1-second system budget.
