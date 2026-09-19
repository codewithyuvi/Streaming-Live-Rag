import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# Use exact relative imports to map correctly
from telemetry.schema import TelemetryEvent, RetrievalEvent, LatenciesMs, TokenCost

from google import genai
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Streaming Live RAG - Phase 1 Baseline")

# Initialize clients globally
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class TurnRequest(BaseModel):
    session_id: str
    turn_id: int
    utterance: str

class TurnResponse(BaseModel):
    answer: str
    telemetry: TelemetryEvent

@app.post("/turn", response_model=TurnResponse)
def handle_turn(req: TurnRequest):
    start_time = time.time()
    
    # 1. Embed query
    query_vec = list(embedding_model.embed([req.utterance]))[0]
    
    # 2. Search Qdrant (Dense only for Phase 1)
    retrieval_start = time.time()
    try:
        search_result = qdrant_client.search(
            collection_name="dev_corpus_dense",
            query_vector=query_vec.tolist(),
            limit=3
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant Search Error: {str(e)}")
        
    retrieval_latency = (time.time() - retrieval_start) * 1000

    # Format chunks for prompt stuffing
    context_blocks = []
    citations = []
    for hit in search_result:
        chunk_text = hit.payload.get("text", "")
        tag = hit.payload.get("tag", "")
        context_blocks.append(f"[{tag}]\n{chunk_text}")
        citations.append(tag)
        
    context_str = "\n\n".join(context_blocks)

    # 3. Naive prompt stuffing
    prompt = f"""
    You are a helpful assistant answering based ONLY on the provided context.
    
    Context:
    {context_str}
    
    User Query:
    {req.utterance}
    
    Please provide an answer. You must cite your claims using the [Doc_XX §Y] tags from the context.
    """
    
    # 4. LLM Call
    llm_start = time.time()
    try:
        response = gemini_client.models.generate_content(
            model=os.getenv("SYNTHESIS_LLM_MODEL", "gemini-3.8-pro"),
            contents=prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Generation Error: {str(e)}")
        
    ttft = (time.time() - llm_start) * 1000
    answer_text = response.text
    
    end_time = time.time()
    total_latency = (end_time - start_time) * 1000
    
    # 5. Build Telemetry
    telemetry = TelemetryEvent(
        session_id=req.session_id,
        turn_id=req.turn_id,
        retrieval_events=[
            RetrievalEvent(
                timestamp_s=round(end_time, 2),
                query=req.utterance,
                trigger="baseline_turn"
            )
        ],
        sub_queries=[req.utterance],
        answer=answer_text,
        citations=citations,
        latencies_ms=LatenciesMs(
            retrieval=round(retrieval_latency, 2),
            time_to_first_token=round(ttft, 2),
            end_to_end=round(total_latency, 2)
        )
    )
    
    return TurnResponse(
        answer=answer_text,
        telemetry=telemetry
    )
