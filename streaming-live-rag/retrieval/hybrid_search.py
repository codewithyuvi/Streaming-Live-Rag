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
                use_local = os.getenv("QDRANT_LOCAL", "").lower() in ("true", "1")
                if not use_local:
                    try:
                        client = QdrantClient(
                            url=qdrant_url,
                            api_key=os.getenv("QDRANT_API_KEY") or None,
                            timeout=int(os.getenv("QDRANT_TIMEOUT_S", "2")),
                            check_compatibility=False,
                        )
                        client.get_collections()
                        _qdrant_client = client
                        return _qdrant_client
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).info(
                            "Remote Qdrant at %s unreachable; using embedded local storage in 'data/qdrant_storage'.",
                            qdrant_url
                        )
                storage_dir = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "data", "qdrant_storage"
                )
                os.makedirs(storage_dir, exist_ok=True)
                _qdrant_client = QdrantClient(path=storage_dir, check_compatibility=False)
    return _qdrant_client


_gpu_info: dict | None = None


def get_gpu_info() -> dict:
    """
    Auto-detects available GPU hardware (NVIDIA CUDA, DirectML, Apple MPS, ROCm).
    Configures ONNX Runtime and PyTorch dynamic libraries so models seamlessly
    bind to the GPU whenever available.
    """
    global _gpu_info
    if _gpu_info is not None:
        return _gpu_info

    info = {
        "gpu_available": False,
        "device_name": "CPU",
        "provider": "CPUExecutionProvider",
        "vram_gb": 0.0,
        "cuda_version": None,
    }

    # 1. Probe PyTorch CUDA & configure Windows DLL path if available
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if hasattr(os, "add_dll_directory") and os.path.isdir(torch_lib):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    # 2. Probe ONNX Runtime Execution Providers
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            info["gpu_available"] = True
            info["provider"] = "CUDAExecutionProvider"
        elif "DmlExecutionProvider" in available:
            info["gpu_available"] = True
            info["provider"] = "DmlExecutionProvider"
        elif "ROCMExecutionProvider" in available:
            info["gpu_available"] = True
            info["provider"] = "ROCMExecutionProvider"
    except Exception:
        pass

    _gpu_info = info
    return _gpu_info


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                gpu = get_gpu_info()
                use_cuda = gpu["gpu_available"] and gpu["provider"] == "CUDAExecutionProvider"
                try:
                    if use_cuda:
                        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cuda=True)
                    else:
                        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                except Exception:
                    _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cuda=False)
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
                gpu = get_gpu_info()
                use_cuda = gpu["gpu_available"] and gpu["provider"] == "CUDAExecutionProvider"
                try:
                    if use_cuda:
                        _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cuda=True)
                    else:
                        _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
                except Exception:
                    _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cuda=False)
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

        t_retrieval_start = time.perf_counter()
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
        # Maintain strict 1:1 alignment: filter points and text payloads in a single pass
        valid_points = [p for p in points if p is not None and getattr(p, "payload", None) is not None]
        docs = [(p.payload or {}).get("text", "") for p in valid_points]

        if not docs:
            return []

        t_rerank_start = time.perf_counter()
        scores = list(cross_encoder.rerank(rerank_query, docs))
        scored_hits = sorted(zip(scores, valid_points), key=lambda x: x[0], reverse=True)
        return scored_hits[:top_k]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Vector search failed for query %r: %s", query[:120], e)
        raise RetrievalError(f"Vector search failed: {e}") from e


def search(query: str, top_k: int = 3) -> Tuple[List[Any], float, float]:
    """
    Convenience wrapper returning (best_points, retrieval_latency_ms, rerank_latency_ms)
    with genuine, independently measured component latencies.
    """
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    sparse_model = get_sparse_embedding_model()
    cross_encoder = get_reranker()

    t_retrieval_start = time.perf_counter()
    query_dense = next(iter(embedding_model.query_embed([query])))
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
    valid_points = [p for p in points if p is not None and getattr(p, "payload", None) is not None]
    docs = [(p.payload or {}).get("text", "") for p in valid_points]
    retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000

    if not docs:
        return [], retrieval_latency_ms, 0.0

    t_rerank_start = time.perf_counter()
    scores = list(cross_encoder.rerank(query, docs))
    scored_hits = sorted(zip(scores, valid_points), key=lambda x: x[0], reverse=True)
    rerank_latency_ms = (time.perf_counter() - t_rerank_start) * 1000

    return [p for _, p in scored_hits[:top_k]], round(retrieval_latency_ms, 2), round(rerank_latency_ms, 2)
