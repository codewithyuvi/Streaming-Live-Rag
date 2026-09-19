import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# Use exact relative imports to map correctly
from telemetry.schema import TelemetryEvent, RetrievalEvent, LatenciesMs, TokenCost, ControllerDecision

from google import genai
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from streaming.stream_simulator import simulate_stream
from controller.heuristics import is_stable_enough
from controller.decide import decide_retrieval

load_dotenv()

app = FastAPI(title="Streaming Live RAG - Phase 1 Baseline")

# Initialize clients globally
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Import Qdrant classes needed for hybrid search
from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion

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
    
    # Simulate streaming
    chunks = list(simulate_stream(req.utterance, words_per_chunk=2, ms_per_chunk=300))
    
    final_query = req.utterance
    controller_decision_log = None
    
    for chunk in chunks:
        # 1. Heuristics filter
        if not is_stable_enough(chunk.partial_text):
            continue
            
        # 2. LLM Controller
        decision_dict = decide_retrieval(chunk.partial_text)
        
        controller_decision_log = ControllerDecision(
            trigger=decision_dict["trigger"],
            timestamp_s=chunk.t_offset_s,
            reason=decision_dict.get("reason", "")
        )
        
        if decision_dict["trigger"] == "retrieve_now":
            final_query = chunk.partial_text
            break
        elif decision_dict["trigger"] == "no_retrieval_needed":
            # Fast-path for greetings etc.
            # For this Phase, we'll just break and still execute the rest to see what happens, or skip Qdrant.
            final_query = chunk.partial_text
            break
            
    # If no decision forced an early break, final_query defaults to the full req.utterance
    
    # 3. Embed final query
    query_dense = list(embedding_model.embed([final_query]))[0]
    query_sparse_obj = list(sparse_embedding_model.embed([final_query]))[0]
    query_sparse = SparseVector(
        indices=query_sparse_obj.indices.tolist(),
        values=query_sparse_obj.values.tolist()
    )
    
    # 2. Search Qdrant (Hybrid RRF)
    retrieval_start = time.time()
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant Search Error: {str(e)}")
        
    retrieval_latency = (time.time() - retrieval_start) * 1000

    # 3. Rerank with Cross-Encoder
    points = search_result.points if hasattr(search_result, "points") else search_result
    
    docs = [hit.payload.get("text", "") for hit in points]
    
    rerank_start = time.time()
    try:
        scores = list(reranker.rerank(req.utterance, docs))
        scored_hits = list(zip(scores, points))
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        best_points = [hit for score, hit in scored_hits[:3]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking Error: {str(e)}")
    
    rerank_latency = (time.time() - rerank_start) * 1000

    # Format chunks for prompt stuffing
    context_blocks = []
    citations = []
    
    for hit in best_points:
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
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=os.getenv("SYNTHESIS_LLM_MODEL", "gemini-3.8-flash"),
                contents=prompt,
            )
            break
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Wait 1s, then 2s before retrying
                continue
            raise HTTPException(status_code=500, detail=f"Gemini Generation Error: {str(e)}")
        
    ttft = (time.time() - llm_start) * 1000
    answer_text = response.text
    
    end_time = time.time()
    total_latency = (end_time - start_time) * 1000
    
    # 5. Build Telemetry
    telemetry = TelemetryEvent(
        session_id=req.session_id,
        turn_id=req.turn_id,
        controller_decision=controller_decision_log,
        retrieval_events=[
            RetrievalEvent(
                timestamp_s=round(end_time, 2),
                query=final_query,
                trigger=controller_decision_log.trigger if controller_decision_log else "baseline_turn"
            )
        ],
        sub_queries=[final_query],
        answer=answer_text,
        citations=citations,
        latencies_ms=LatenciesMs(
            retrieval=round(retrieval_latency, 2),
            rerank=round(rerank_latency, 2),
            time_to_first_token=round(ttft, 2),
            end_to_end=round(total_latency, 2)
        )
    )
    
    return TurnResponse(
        answer=answer_text,
        telemetry=telemetry
    )
