import os
import threading
import time
from typing import List, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from dotenv import load_dotenv

load_dotenv()


def _collection_name() -> str:
    # Read at call time (not import time) so tests / compose overrides apply.
    return os.getenv("COLLECTION_NAME", "dev_corpus_dense")


# Backward-compat alias (was a frozen module constant).
COLLECTION_NAME = _collection_name()

# Cached model and client instances
_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()
_embedding_model: TextEmbedding | None = None
_sparse_embedding_model: SparseTextEmbedding | None = None
_reranker: TextCrossEncoder | None = None
_model_lock = threading.Lock()


class RetrievalError(RuntimeError):
    """Raised when Qdrant / embedding / rerank fails (distinct from 'no hits')."""


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                qdrant_url = os.getenv("QDRANT_URL") or f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
                _qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=os.getenv("QDRANT_API_KEY") or None,
                    timeout=int(os.getenv("QDRANT_TIMEOUT_S", "10")),
                )
    return _qdrant_client


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model


def get_sparse_embedding_model() -> SparseTextEmbedding:
    global _sparse_embedding_model
    if _sparse_embedding_model is None:
        with _model_lock:
            if _sparse_embedding_model is None:
                _sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_embedding_model


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve_and_rerank(query: str, rerank_against: str | None = None, top_k: int = 5) -> List[Tuple[float, Any]]:
    """
    Executes hybrid RRF search (dense + sparse with query_embed) and reranks with cross-encoder.
    Returns list of (score, point) tuples sorted by score descending.
    Raises RetrievalError on backend failure (callers decide degraded vs 504).
    """
    if not (query or "").strip():
        return []
    top_k = max(1, min(int(top_k or 5), 20))
    client = get_qdrant_client()
    emb_model = get_embedding_model()
    sparse_model = get_sparse_embedding_model()
    cross_encoder = get_reranker()

    rerank_query = rerank_against or query

    try:
        # Use query_embed instead of document embed
        query_dense = next(iter(emb_model.query_embed([query])))
        query_sparse_obj = next(iter(sparse_model.query_embed([query])))
        query_sparse = SparseVector(
            indices=query_sparse_obj.indices.tolist(),
            values=query_sparse_obj.values.tolist()
        )

        prefetch_limit = max(top_k * 2, 10)
        search_result = client.query_points(
            collection_name=_collection_name(),
            prefetch=[
                Prefetch(query=query_dense.tolist(), using="dense", limit=prefetch_limit),
                Prefetch(query=query_sparse, using="sparse", limit=prefetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=max(top_k * 2, 10),
        )

        points = search_result.points if hasattr(search_result, "points") else search_result
        docs = [(hit.payload or {}).get("text", "") for hit in points if hit is not None and getattr(hit, "payload", None) is not None]

        if not docs:
            return []

        scores = list(cross_encoder.rerank(rerank_query, docs))
        scored_hits = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)
        return scored_hits[:top_k]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Vector search failed for query %r: %s", query[:120], e)
        raise RetrievalError(f"Vector search failed: {e}") from e


def search(query: str, top_k: int = 3) -> Tuple[List[Any], float, float]:
    """
    Convenience wrapper returning (best_points, retrieval_latency_ms, rerank_latency_ms).
    """
    t0 = time.time()
    scored_hits = retrieve_and_rerank(query, top_k=top_k)
    t_end = time.time()
    total_ms = (t_end - t0) * 1000
    points = [hit for _, hit in scored_hits]
    return points, total_ms * 0.6, total_ms * 0.4
