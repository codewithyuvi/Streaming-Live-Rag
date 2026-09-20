import yaml
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Setup clients (same logic as main.py but for benchmark)
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from retrieval.hybrid_search import search as hybrid_search

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_dense_search(query: str, top_k: int = 3):
    start = time.time()
    query_dense = list(embedding_model.embed([query]))[0]
    results = qdrant_client.query_points(
        collection_name="dev_corpus_dense",
        query=query_dense.tolist(),
        using="dense",
        limit=top_k
    )
    latency = (time.time() - start) * 1000
    # query_points returns a QueryResponse, so we can access .points
    points = results.points if hasattr(results, "points") else results
    return points, latency

def main():
    with open("eval/labeled_set.yaml", "r") as f:
        data = yaml.safe_load(f)
    
    queries = [q for q in data["queries"] if q["expected_doc"]]
    
    dense_hits = 0
    hybrid_hits = 0
    total_dense_latency = 0
    total_hybrid_latency = 0
    total_queries = len(queries)
    
    print("=== Retrieval Benchmark (Dense vs Hybrid) ===")
    
    for q in queries:
        expected_docs = [d.strip() for d in q["expected_doc"].split(",")]
        query_text = q["utterance"]
        
        # Dense
        dense_results, dense_latency = run_dense_search(query_text)
        total_dense_latency += dense_latency
        dense_tags = [hit.payload.get("tag", "").split(" ")[0] for hit in dense_results]
        # Check if ANY expected doc is in top K
        if any(doc in dense_tags for doc in expected_docs):
            dense_hits += 1
            
        # Hybrid
        hybrid_results, h_latency, r_latency = hybrid_search(query_text, top_k=3)
        hybrid_total_latency = h_latency + r_latency
        total_hybrid_latency += hybrid_total_latency
        hybrid_tags = [hit.payload.get("tag", "").split(" ")[0] for hit in hybrid_results]
        if any(doc in hybrid_tags for doc in expected_docs):
            hybrid_hits += 1
            
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Dense Recall@3: {dense_hits / total_queries * 100:.2f}%")
    print(f"Hybrid Recall@3: {hybrid_hits / total_queries * 100:.2f}%")
    print(f"Average Dense Latency: {total_dense_latency / total_queries:.2f}ms")
    print(f"Average Hybrid Latency: {total_hybrid_latency / total_queries:.2f}ms")

if __name__ == "__main__":
    main()
