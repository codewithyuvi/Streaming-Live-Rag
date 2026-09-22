import os
import glob
import logging
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector, Modifier

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dev_corpus_dense")


def parse_corpus(corpus_dir: str):
    """
    Parses all .txt files in the given directory.
    Supports:
    1. Pre-tagged format:
       Doc_01 §1
       [Chunk text...]
    2. Fallback windowed chunking with synthetic tags if header missing.
    """
    chunks = []
    txt_files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    
    for doc_idx, file_path in enumerate(txt_files, start=1):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                continue
            
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
            sec_idx = 1
            for block in blocks:
                parts = block.split("\n", 1)
                if len(parts) == 2 and "§" in parts[0]:
                    tag = parts[0].strip()
                    text = parts[1].strip()
                    tag_parts = tag.split()
                    doc_id = tag_parts[0] if tag_parts else f"Doc_{doc_idx:02d}"
                    section = tag_parts[1].lstrip("§") if len(tag_parts) > 1 else str(sec_idx)
                else:
                    # Fallback for untagged text
                    doc_id = f"Doc_{doc_idx:02d}"
                    section = str(sec_idx)
                    tag = f"{doc_id} §{section}"
                    text = block.strip()
                    logger.warning(f"Untagged block in {file_path}, assigned synthetic tag {tag}")

                chunks.append({
                    "doc_id": doc_id,
                    "section": section,
                    "tag": tag,
                    "text": text
                })
                sec_idx += 1
                
    return chunks


def ingest():
    corpus_dir = os.path.join(os.path.dirname(__file__), "../data/dev_corpus")
    print(f"Parsing corpus from {corpus_dir}...")
    chunks = parse_corpus(corpus_dir)
    print(f"Found {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found. Exiting.")
        return

    # Calculate average chunk length for BM25 calibration
    avg_len = sum(len(c["text"].split()) for c in chunks) / max(len(chunks), 1)
    print(f"Calibrated BM25 avg_len: {avg_len:.2f}")

    # Initialize FastEmbed for dense vectors
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Initialize FastEmbed for sparse vectors with IDF and calibrated avg_len
    print("Loading sparse embedding model (Qdrant/bm25)...")
    sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25", avg_len=avg_len)
    
    texts = [c["text"] for c in chunks]
    print("Generating dense embeddings...")
    embeddings = list(embedding_model.embed(texts))
    
    print("Generating sparse embeddings...")
    sparse_embeddings = list(sparse_embedding_model.embed(texts))

    # Initialize Qdrant Client
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    
    collection_name = COLLECTION_NAME
    
    # Idempotent collection creation
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' exists. Recreating it to ensure clean state...")
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF)
        }
    )
    
    print("Upserting vectors to Qdrant...")
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i,
                vector={
                    "dense": embeddings[i].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_embeddings[i].indices.tolist(),
                        values=sparse_embeddings[i].values.tolist()
                    )
                },
                payload={
                    "doc_id": chunk["doc_id"],
                    "section": chunk["section"],
                    "tag": chunk["tag"],
                    "text": chunk["text"]
                }
            )
        )
        
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Ingestion complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest()
