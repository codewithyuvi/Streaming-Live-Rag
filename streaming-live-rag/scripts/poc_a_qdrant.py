import asyncio
from qdrant_client import AsyncQdrantClient, models

async def main():
    print("Connecting to local Qdrant...")
    client = AsyncQdrantClient(url="http://localhost:6333")
    
    collection_name = "poc_collection"
    
    # 1. Create Collection (using dense + sparse)
    print("Creating collection with Dense and Sparse vectors...")
    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)
        
    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )
    
    # NOTE: To complete this POC (as per guide), you will need to:
    # 2. Use fastembed to generate dense (bge-small) and sparse (bm25) embeddings for data/dev_corpus
    # 3. Insert them into Qdrant using client.upsert()
    # 4. Perform a hybrid search using prefetch(dense) + prefetch(sparse) with RRF fusion.
    # 5. Measure the latency.
    
    print("Qdrant connection successful. Collection created.")
    print("Next step: Implement fastembed generation and RRF fusion.")

if __name__ == "__main__":
    asyncio.run(main())
