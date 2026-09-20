import os
import asyncio
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from telemetry.schema import TelemetryEvent, RetrievalEvent, LatenciesMs, TokenCost, ControllerDecision

from google import genai
from dotenv import load_dotenv
from streaming.stream_simulator import simulate_stream
from controller.heuristics import is_stable_enough
from controller.decide import decide_retrieval
from controller.decompose import decompose_query
from controller.refinement import classify_refinement
from retrieval.merge import merge_and_dedup
from retrieval.grounding import validate_grounding
from session.store import get_or_create_session, TurnRecord
from retrieval.hybrid_search import search
from telemetry.sink import emit

load_dotenv()

app = FastAPI(title="Streaming Live RAG - Phase 5 Session-Aware")

# Initialize LLM client globally
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion


class TurnRequest(BaseModel):
    session_id: str
    turn_id: int
    utterance: str


class TurnResponse(BaseModel):
    answer: str
    telemetry: TelemetryEvent


# ---------------------------------------------------------------------------
# Helpers — retrieval for a single (sub-)query
# ---------------------------------------------------------------------------

def _retrieve_and_rerank_single(query_text: str, rerank_against: str, top_k: int = 5):
    """
    Runs hybrid RRF retrieval + cross-encoder rerank for one query string.
    Returns list of (score, point) tuples sorted by rerank score descending.
    """
    query_dense = list(embedding_model.embed([query_text]))[0]
    query_sparse_obj = list(sparse_embedding_model.embed([query_text]))[0]
    query_sparse = SparseVector(
        indices=query_sparse_obj.indices.tolist(),
        values=query_sparse_obj.values.tolist()
    )

    search_result = qdrant_client.query_points(
        collection_name="dev_corpus_dense",
        prefetch=[
            Prefetch(query=query_dense.tolist(), using="dense", limit=10),
            Prefetch(query=query_sparse, using="sparse", limit=10),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )

    points = search_result.points if hasattr(search_result, "points") else search_result
    docs = [hit.payload.get("text", "") for hit in points]

    scores = list(reranker.rerank(rerank_against, docs))
    scored_hits = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)
    return scored_hits


async def _async_retrieve_single(query_text: str, rerank_against: str, top_k: int = 5):
    """Wraps synchronous retrieval in a thread for asyncio.gather concurrency."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _retrieve_and_rerank_single, query_text, rerank_against, top_k
    )


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/turn", response_model=TurnResponse)
async def handle_turn(req: TurnRequest):
    start_time = time.time()

    # ── Session management ─────────────────────────────────────────────
    session = get_or_create_session(req.session_id)
    last_turn = session.get_last_turn()
    history_context = session.get_history_context(max_turns=5)

    # ── Phase 5: Refinement classification ─────────────────────────────
    refinement_start = time.time()
    refinement_result = classify_refinement(
        current_utterance=req.utterance,
        conversation_history=history_context,
        previous_answer=last_turn.answer if last_turn else "",
    )
    refinement_type = refinement_result["type"]
    refinement_latency = (time.time() - refinement_start) * 1000

    # ── PRESENTATION_ONLY: skip everything ─────────────────────────────
    if refinement_type == "PRESENTATION_ONLY":
        answer_text = _presentation_response(req.utterance)

        session.add_turn(TurnRecord(
            turn_id=req.turn_id,
            utterance=req.utterance,
            answer=answer_text,
            citations=[],
            sub_queries=[],
            refinement_type="PRESENTATION_ONLY",
        ))

        telemetry = TelemetryEvent(
            session_id=req.session_id,
            turn_id=req.turn_id,
            refinement_type="PRESENTATION_ONLY",
            answer=answer_text,
            answer_version=session.answer_version,
            latencies_ms=LatenciesMs(
                refinement=round(refinement_latency, 2),
                end_to_end=round((time.time() - start_time) * 1000, 2),
            ),
        )
        return TurnResponse(answer=answer_text, telemetry=telemetry)

    # ── Phase 3: Streaming controller ──────────────────────────────────
    chunks = list(simulate_stream(req.utterance, words_per_chunk=2, ms_per_chunk=300))
    final_query = req.utterance
    controller_decision_log = None

    for chunk in chunks:
        if not is_stable_enough(chunk.partial_text):
            continue

        decision_dict = decide_retrieval(chunk.partial_text)
        controller_decision_log = ControllerDecision(
            trigger=decision_dict["trigger"],
            timestamp_s=chunk.t_offset_s,
            reason=decision_dict.get("reason", ""),
        )

        if decision_dict["trigger"] == "retrieve_now":
            final_query = chunk.partial_text
            break
        elif decision_dict["trigger"] == "no_retrieval_needed":
            final_query = chunk.partial_text
            break

    # ── Phase 4: Multi-Intent Decomposition ────────────────────────────
    decompose_start = time.time()
    sub_query_dicts = decompose_query(final_query)
    decompose_latency = (time.time() - decompose_start) * 1000

    sub_query_texts = [sq["sub_query"] for sq in sub_query_dicts]
    is_multi_intent = len(sub_query_texts) > 1

    # ── Parallel Retrieval via asyncio.gather ──────────────────────────
    retrieval_start = time.time()
    try:
        if is_multi_intent:
            tasks = [
                _async_retrieve_single(sq, sq, top_k=5) for sq in sub_query_texts
            ]
            all_scored_hits = await asyncio.gather(*tasks)
        else:
            query_to_search = sub_query_texts[0] if sub_query_texts else final_query
            all_scored_hits = [
                _retrieve_and_rerank_single(query_to_search, query_to_search, top_k=5)
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval Error: {str(e)}")

    retrieval_latency = (time.time() - retrieval_start) * 1000

    # ── Merge & Dedup across sub-queries ──────────────────────────────
    per_subquery_results = []
    for i, scored_hits in enumerate(all_scored_hits):
        sq_text = sub_query_texts[i] if i < len(sub_query_texts) else final_query
        per_subquery_results.append({
            "sub_query": sq_text,
            "scored_hits": scored_hits,
        })

    merged_chunks = merge_and_dedup(per_subquery_results, top_k=5)

    # ── Build context for synthesis ───────────────────────────────────
    context_blocks = []
    citations = []
    for mc in merged_chunks:
        source_info = ""
        if is_multi_intent and mc.source_sub_queries:
            source_info = f"  (relevant to: {', '.join(mc.source_sub_queries)})"
        context_blocks.append(f"[{mc.tag}]{source_info}\n{mc.text}")
        citations.append(mc.tag)

    context_str = "\n\n".join(context_blocks)

    # ── Phase 5: Session-aware synthesis prompt ───────────────────────
    prompt = _build_synthesis_prompt(
        utterance=req.utterance,
        context_str=context_str,
        is_multi_intent=is_multi_intent,
        sub_query_texts=sub_query_texts,
        refinement_type=refinement_type,
        refinement_constraint=refinement_result.get("constraint", ""),
        previous_answer=last_turn.answer if last_turn else "",
        history_context=history_context,
    )

    # ── LLM Synthesis ─────────────────────────────────────────────────
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
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Gemini Generation Error: {str(e)}",
            )

    ttft = (time.time() - llm_start) * 1000
    answer_text = response.text

    # ── Phase 5: Grounding validation ─────────────────────────────────
    grounding_start = time.time()
    grounding_report = validate_grounding(answer_text, citations)
    grounding_latency = (time.time() - grounding_start) * 1000

    # Determine uncertainty text
    uncertainty_text = ""
    if grounding_report.fabricated_tags:
        uncertainty_text = f"Warning: fabricated citations detected: {grounding_report.fabricated_tags}"
    elif not grounding_report.is_grounded and not grounding_report.uncertainty_expressed:
        uncertainty_text = "Warning: answer may not be fully grounded in corpus"

    end_time = time.time()
    total_latency = (end_time - start_time) * 1000

    # ── Update session state ──────────────────────────────────────────
    session.add_turn(TurnRecord(
        turn_id=req.turn_id,
        utterance=req.utterance,
        answer=answer_text,
        citations=citations,
        sub_queries=sub_query_texts,
        refinement_type=refinement_type,
    ))

    # ── Telemetry ─────────────────────────────────────────────────────
    retrieval_events = []
    for sq in sub_query_texts:
        retrieval_events.append(
            RetrievalEvent(
                timestamp_s=round(time.time(), 2),
                query=sq,
                trigger=controller_decision_log.trigger if controller_decision_log else "decomposed",
            )
        )

    telemetry = TelemetryEvent(
        session_id=req.session_id,
        turn_id=req.turn_id,
        controller_decision=controller_decision_log,
        retrieval_events=[
            RetrievalEvent(
                timestamp_s=controller_decision_log.timestamp_s if controller_decision_log else 0.0,
                query=final_query,
                trigger=controller_decision_log.trigger if controller_decision_log else "baseline_turn"
            )
        ],
        sub_queries=[final_query],
        refinement_type=refinement_type,
        retrieval_events=retrieval_events,
        sub_queries=sub_query_texts,
        answer=answer_text,
        citations=citations,
        uncertainty=uncertainty_text,
        grounding_score=grounding_report.score,
        grounding_report=grounding_report.to_dict(),
        answer_version=session.answer_version,
        latencies_ms=LatenciesMs(
            retrieval=round(retrieval_latency, 2),
            rerank=0.0,
            decompose=round(decompose_latency, 2),
            refinement=round(refinement_latency, 2),
            grounding=round(grounding_latency, 2),
            time_to_first_token=round(ttft, 2),
            end_to_end=round(total_latency, 2),
        ),
    )

    return TurnResponse(answer=answer_text, telemetry=telemetry)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _presentation_response(utterance: str) -> str:
    """Quick response for presentation-only turns (no retrieval needed)."""
    u = utterance.strip().lower()
    if any(w in u for w in ["thank", "thanks", "thx"]):
        return "You're welcome! Let me know if you have any other questions."
    if any(w in u for w in ["ok", "okay", "got it", "understood", "i see"]):
        return "Understood. Feel free to ask anything else."
    if any(w in u for w in ["great", "perfect", "awesome", "nice"]):
        return "Glad that helps! Is there anything else you'd like to know?"
    return "I'm here to help. Please go ahead with your question."


def _build_synthesis_prompt(
    utterance: str,
    context_str: str,
    is_multi_intent: bool,
    sub_query_texts: list[str],
    refinement_type: str,
    refinement_constraint: str,
    previous_answer: str,
    history_context: str,
) -> str:
    """Builds the full synthesis prompt based on refinement type and intent count."""

    # Base grounding instructions (always present)
    grounding_rules = """
STRICT GROUNDING RULES:
1. Every factual claim MUST be cited with [Doc_XX §Y] tags from the context.
2. Do NOT invent or fabricate any Doc IDs or Section numbers.
3. If the context does not contain information to answer a question, you MUST explicitly state:
   "This information is not available in the provided documents."
4. Never answer from your own knowledge. Only use the provided context.
5. If uncertain, express uncertainty explicitly.
"""

    if refinement_type == "LATE_DETAIL":
        # Refinement mode: update previous answer with new constraint
        return f"""You are a helpful assistant answering based ONLY on the provided context.

The user previously asked a question and you provided an answer. Now they are adding
a new constraint or clarification. You must REFINE your previous answer — do not start
from scratch. Selectively update only the parts affected by the new detail.

Previous Answer:
{previous_answer[:500]}

New Constraint from User:
{utterance}
(Detected constraint: {refinement_constraint})

Context (retrieved passages):
{context_str}

{grounding_rules}

Provide a refined answer that incorporates the new constraint while preserving
the valid parts of the previous answer. Cite all claims.
"""

    if history_context:
        history_section = f"""
Conversation History (for context only — do NOT repeat previous answers):
{history_context}

"""
    else:
        history_section = ""

    if is_multi_intent:
        sub_query_list = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_query_texts))
        return f"""You are a helpful assistant answering based ONLY on the provided context.
{history_section}
The user's question contains multiple distinct information needs.
The decomposed sub-questions are:
{sub_query_list}

Context (retrieved passages):
{context_str}

User Query:
{utterance}

{grounding_rules}

Instructions:
1. Answer EACH sub-question separately, using information ONLY from the context.
2. Cite every factual claim using the [Doc_XX §Y] tags from the context.
3. If the context does not contain information for a sub-question, explicitly state
   that the information is not available in the provided documents.
4. Structure your response clearly, addressing each sub-question in order.
"""
    else:
        return f"""You are a helpful assistant answering based ONLY on the provided context.
{history_section}
Context:
{context_str}

User Query:
{utterance}

{grounding_rules}

Provide a clear, well-cited answer.
"""

