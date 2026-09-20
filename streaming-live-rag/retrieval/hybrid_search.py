import os
import time
from typing import List, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Initialize clients globally so they are cached
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

def search(query: str, top_k: int = 3) -> Tuple[List[Any], float, float]:
    """
    Executes a hybrid search in Qdrant (Dense + Sparse with RRF),
    then reranks the top candidates using a Cross-Encoder.
    
    Returns:
        (best_points, retrieval_latency_ms, rerank_latency_ms)
    """
    # 1. Embed final query
    query_dense = list(embedding_model.embed([query]))[0]
    query_sparse_obj = list(sparse_embedding_model.embed([query]))[0]
    query_sparse = SparseVector(
        indices=query_sparse_obj.indices.tolist(),
        values=query_sparse_obj.values.tolist()
    )
    
    # 2. Search Qdrant (Hybrid RRF)
    retrieval_start = time.time()
    search_result = qdrant_client.query_points(
        collection_name="dev_corpus_dense",
        prefetch=[
            Prefetch(
                query=query_dense.tolist(),
                using="dense",
                limit=10,
            ),
            Prefetch(
                query=query_sparse,
                using="sparse",
                limit=10,
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5
    )
    retrieval_latency = (time.time() - retrieval_start) * 1000

    # 3. Rerank with Cross-Encoder
    points = search_result.points if hasattr(search_result, "points") else search_result
    docs = [hit.payload.get("text", "") for hit in points]
    
    rerank_start = time.time()
    if not docs:
        return [], retrieval_latency, 0.0
        
    scores = list(reranker.rerank(query, docs))
    scored_hits = list(zip(scores, points))
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    best_points = [hit for score, hit in scored_hits[:top_k]]
    
    rerank_latency = (time.time() - rerank_start) * 1000
    
    return best_points, retrieval_latency, rerank_latency
