import os
import glob
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def parse_corpus(corpus_dir: str):
    """
    Parses all .txt files in the given directory.
    Expected format in txt files:
    Doc_01 §1
    [Chunk text...]
    """
    chunks = []
    
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n\n") # split by paragraphs
            
            for block in lines:
                block = block.strip()
                if not block:
                    continue
                
                parts = block.split("\n", 1)
                if len(parts) == 2 and "§" in parts[0]:
                    tag = parts[0].strip()
                    text = parts[1].strip()
                    doc_id = tag.split(" ")[0]
                    section = tag.split(" ")[1]
                    
                    chunks.append({
                        "doc_id": doc_id,
                        "section": section,
                        "tag": tag,
                        "text": text
                    })
    return chunks

def ingest():
    corpus_dir = os.path.join(os.path.dirname(__file__), "../data/dev_corpus")
    print(f"Parsing corpus from {corpus_dir}...")
    chunks = parse_corpus(corpus_dir)
    print(f"Found {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found. Exiting.")
        return

    # Initialize FastEmbed for dense vectors
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    texts = [c["text"] for c in chunks]
    print("Generating dense embeddings...")
    embeddings = list(embedding_model.embed(texts))

    # Initialize Qdrant Client
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    
    collection_name = "dev_corpus_dense"
    
    # Idempotent collection creation
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' exists. Recreating it to ensure clean state...")
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    print("Upserting vectors to Qdrant...")
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
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
