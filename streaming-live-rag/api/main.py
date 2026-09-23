import os
import sys
import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telemetry.schema import (
    TelemetryEvent,
    RetrievalEvent,
    LatenciesMs,
    TokenCost,
    ControllerDecision,
)
from dotenv import load_dotenv

from streaming.stream_simulator import simulate_stream
from controller.heuristics import is_stable_enough
from controller.decide import decide_retrieval
from controller.decompose import decompose_query
from controller.refinement import classify_refinement
from retrieval.merge import merge_and_dedup
from retrieval.grounding import validate
from session.store import get_or_create_session
from retrieval.hybrid_search import retrieve_and_rerank
from telemetry.sink import emit

load_dotenv()

app = FastAPI(title="Streaming Live RAG - Live Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

_gemini_client = None
_gemini_lock = threading.Lock()


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                try:
                    from google import genai
                except ImportError:
                    raise ImportError("The 'google-genai' package is not installed. Please run: pip install google-genai")
                api_key = os.getenv("GEMINI_API_KEY", "")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY is not set in environment.")
                _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


SYNTHESIS_MODEL = os.getenv("SYNTHESIS_LLM_MODEL", "gemini-2.5-flash")

# In-memory per-session rate limiter (H2)
_session_request_times: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()
RATE_LIMIT_PER_MINUTE = 60


def _check_rate_limit(session_id: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        timestamps = _session_request_times.setdefault(session_id, [])
        valid = [t for t in timestamps if now - t < 60.0]
        if len(valid) >= RATE_LIMIT_PER_MINUTE:
            return False
        valid.append(now)
        _session_request_times[session_id] = valid
        return True


@app.get("/health")
def health_check(live: bool = False):
    """Healthcheck endpoint for Docker container and reproducibility validation (Gate G1)."""
    res = {"status": "ok", "version": "0.1.0"}
    if live:
        res["providers"] = {
            "groq": "configured" if bool(os.getenv("GROQ_API_KEY")) else "missing",
            "gemini": "configured" if bool(os.getenv("GEMINI_API_KEY")) else "missing",
        }
    return res


@app.get("/")
@app.get("/demo")
def demo_ui():
    """Serves the interactive live streaming demo dashboard."""
    static_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "index.html"))
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Streaming Live RAG API is running. Access /turn for API queries."}


class TurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$")
    turn_id: int = Field(..., ge=1, le=10000)
    utterance: str = Field(..., min_length=1, max_length=1000)


class TurnResponse(BaseModel):
    answer: str
    telemetry: TelemetryEvent


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _same_intent(q1: str, q2: str | None) -> bool:
    """Checks if a decomposed sub-query covers the same intent as the provisional query.

    Uses token-overlap (Jaccard) instead of raw substring so a truncated
    provisional prefix does not suppress a needed delta retrieval.
    """
    if not q2:
        return False
    import re

    def _toks(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.strip().lower()))

    t1, t2 = _toks(q1), _toks(q2)
    if not t1 or not t2:
        return False
    if q1.strip().lower() == q2.strip().lower():
        return True
    inter = len(t1 & t2)
    union = len(t1 | t2)
    jaccard = inter / union if union else 0.0
    # Same intent only if strong overlap AND provisional covers most of the sub-query (t1)
    subquery_recall = inter / len(t1) if t1 else 0.0
    return jaccard >= 0.6 and subquery_recall >= 0.8


def _call_gemini_sync(prompt: str):
    """Executes a synchronous Gemini generate_content call with transient retries."""
    client = get_gemini_client()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=SYNTHESIS_MODEL,
                contents=prompt,
            )
            return response
        except Exception as e:
            err = str(e)
            is_transient = any(c in err for c in ("429", "500", "502", "503", "ResourceExhausted", "timeout"))
            if attempt == max_retries - 1 or not is_transient:
                raise
            time.sleep(2 ** attempt)


def _conversational_reply(utterance: str) -> str:
    """Generates a brief polite response for chit-chat turns without facts or citations."""
    try:
        prompt = (
            "You are a helpful, courteous assistant. Respond briefly and politely to this greeting or remark. "
            "Do NOT reference any documents, citations, or external facts:\n\n"
            f"User: {utterance}"
        )
        res = _call_gemini_sync(prompt)
        return (res.text or "").strip() or "Hello! How can I assist you with the documents today?"
    except Exception:
        return "Hello! How can I assist you with the documents today?"


def _reformat_reply(instruction: str, base_answer: str) -> str:
    """Reformats an existing substantive answer per presentation instruction without new facts."""
    try:
        prompt = f"""You are a helpful assistant. Reformat the following previous answer according to the user instruction:
User Instruction: {instruction}

Previous Answer:
{base_answer}

STRICT RULES:
1. Preserve all factual claims and citation tags ([Doc_XX §Y]) exactly as they appear in the previous answer.
2. Do NOT add new factual claims or fabricate new citation tags.
3. Follow the formatting requested (e.g. bullet points, concise summary, simpler wording).
"""
        res = _call_gemini_sync(prompt)
        text = (res.text or "").strip()
        return text or base_answer
    except Exception:
        return base_answer


# ---------------------------------------------------------------------------
# Main Turn Endpoint
# ---------------------------------------------------------------------------

@app.post("/turn", response_model=TurnResponse)
async def handle_turn(req: TurnRequest):
    if not _check_rate_limit(req.session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for this session (max 60 requests/minute).")
    start_time = time.time()
    session = get_or_create_session(req.session_id)
    has_history = len(session.turns) > 0

    # ── Phase 5: Refinement classification ─────────────────────────────
    refinement_start = time.time()
    if not has_history:
        refinement_result = classify_refinement(
            current_utterance=req.utterance,
            conversation_history="",
            previous_answer="",
        )
    else:
        refinement_result = await asyncio.to_thread(
            classify_refinement,
            current_utterance=req.utterance,
            conversation_history=session.get_history_context(max_turns=5),
            previous_answer=session.current_answer,
        )
    refinement_type = refinement_result.get("type", "NEW_TOPIC")
    refinement_latency = (time.time() - refinement_start) * 1000

    # ── PRESENTATION_ONLY: Reformat or conversational opener (C4) ──────
    if refinement_type == "PRESENTATION_ONLY":
        if session.current_answer:
            # Reformat existing substantive answer
            answer_text = await asyncio.to_thread(_reformat_reply, req.utterance, session.current_answer)
            # Verify no citations were fabricated during reformatting
            check_report = validate(answer_text, session.current_citations)
            if check_report.fabricated:
                answer_text = session.current_answer
            citations = check_report.cited
        else:
            # Conversational opener / chit-chat with no prior answer
            answer_text = await asyncio.to_thread(_conversational_reply, req.utterance)
            citations = []

        session.commit("PRESENTATION_ONLY", req.utterance, answer_text, citations)

        telemetry = TelemetryEvent(
            session_id=req.session_id,
            turn_id=req.turn_id,
            refinement_type="PRESENTATION_ONLY",
            retrieval_required=False,
            retrieval_skip_reason=refinement_result.get("reason", "presentation_only"),
            retrieval_events=[],
            sub_queries=[],
            answer=answer_text,
            citations=citations,
            answer_version=session.answer_version,
            latencies_ms=LatenciesMs(
                refinement=round(refinement_latency, 2),
                controller=round(refinement_latency, 2),
                end_to_end=round((time.time() - start_time) * 1000, 2),
            ),
        )
        emit(telemetry)
        return TurnResponse(answer=answer_text, telemetry=telemetry)

    # ── Phase 3: Two-Stage Streaming Controller (C2 / C3 / H4) ──────────
    chunks = list(simulate_stream(req.utterance, words_per_chunk=2, ms_per_chunk=300))
    t_end = chunks[-1].t_offset_s if chunks else 0.0
    decisions: list[ControllerDecision] = []
    retrieval_events: list[RetrievalEvent] = []
    provisional = None  # (query_text, asyncio.Task, t_offset_s)
    controller_calls = 0
    last_word_count = 0

    for chunk in chunks:
        if not is_stable_enough(chunk.partial_text):
            continue

        words = len(chunk.partial_text.split())
        if (words - last_word_count) < 2 or controller_calls >= 3:
            continue

        if provisional is not None:
            # Early retrieval already dispatched; keep listening to speech
            continue

        d = await asyncio.to_thread(decide_retrieval, chunk.partial_text)
        controller_calls += 1
        last_word_count = words
        decision_record = ControllerDecision(
            trigger=d["trigger"],
            timestamp_s=chunk.t_offset_s,
            reason=d.get("reason", ""),
        )
        decisions.append(decision_record)

        if d["trigger"] == "no_retrieval_needed":
            # Early exit: Chit-chat detected (C2)
            answer_text = await asyncio.to_thread(_conversational_reply, req.utterance)
            session.commit("PRESENTATION_ONLY", req.utterance, answer_text, [])
            telemetry = TelemetryEvent(
                session_id=req.session_id,
                turn_id=req.turn_id,
                controller_decisions=decisions,
                controller_decision=decision_record,
                refinement_type="PRESENTATION_ONLY",
                retrieval_required=False,
                retrieval_skip_reason=d.get("reason", "no_retrieval_needed"),
                retrieval_events=[],
                sub_queries=[],
                answer=answer_text,
                citations=[],
                answer_version=session.answer_version,
                latencies_ms=LatenciesMs(
                    controller=round((time.time() - start_time) * 1000, 2),
                    end_to_end=round((time.time() - start_time) * 1000, 2),
                ),
            )
            emit(telemetry)
            return TurnResponse(answer=answer_text, telemetry=telemetry)

        elif d["trigger"] == "retrieve_now":
            # Early provisional retrieval (C3): Start work without stopping listening!
            prov_query = chunk.partial_text
            prov_task = asyncio.create_task(
                asyncio.to_thread(retrieve_and_rerank, prov_query, prov_query, 5)
            )
            provisional = (prov_query, prov_task, chunk.t_offset_s)
            retrieval_events.append(
                RetrievalEvent(
                    timestamp_s=chunk.t_offset_s,
                    query=prov_query,
                    trigger="provisional",
                )
            )

    # ── Phase 4: Full Utterance Multi-Intent Decomposition (C3 / C7) ───
    decompose_start = time.time()
    if refinement_type == "LATE_DETAIL":
        constraint = refinement_result.get("constraint") or req.utterance
        effective_query = f"{session.current_query}. Additional constraint: {constraint}"
    else:
        effective_query = req.utterance

    sub_query_dicts = await asyncio.to_thread(decompose_query, effective_query)
    decompose_latency = (time.time() - decompose_start) * 1000

    sub_query_texts = [
        sq["sub_query"] for sq in sub_query_dicts if isinstance(sq, dict) and sq.get("sub_query")
    ]
    if not sub_query_texts:
        sub_query_texts = [effective_query]

    # Calculate delta: Sub-queries not already covered by provisional retrieval
    todo = [
        q for q in sub_query_texts
        if not _same_intent(q, provisional[0] if provisional else None)
    ]

    retrieval_start = time.time()
    try:
        delta_tasks = [asyncio.to_thread(retrieve_and_rerank, q, q, 5) for q in todo]
        delta_hits = await asyncio.gather(*delta_tasks) if delta_tasks else []
        prov_hits = (await provisional[1]) if provisional else []
    except Exception:
        delta_hits = []
        prov_hits = []
    retrieval_latency = (time.time() - retrieval_start) * 1000

    # Build per-subquery results structure for merging
    per_subquery_results = []
    if provisional:
        per_subquery_results.append({
            "sub_query": provisional[0],
            "scored_hits": prov_hits,
        })
    for q_text, hits in zip(todo, delta_hits):
        per_subquery_results.append({
            "sub_query": q_text,
            "scored_hits": hits,
        })
        trigger_label = "multi_intent" if len(sub_query_texts) > 1 else ("provisional" if provisional else "end_of_utterance")
        retrieval_events.append(
            RetrievalEvent(
                timestamp_s=t_end,
                query=q_text,
                trigger=trigger_label,
            )
        )

    # ── Quota Merge & Dedup (H8) ───────────────────────────────────────
    merged_chunks = merge_and_dedup(per_subquery_results, top_k=8)

    # ── Relevance Floor Check (C6) ─────────────────────────────────────
    best_score = max((c.best_score for c in merged_chunks), default=-999.0)
    if not merged_chunks or best_score < -15.0:
        # Out-of-corpus / unanswerable request
        answer_text = "This information is not available in the provided documents."
        session.commit(refinement_type, req.utterance, answer_text, [], effective_query=effective_query)
        telemetry = TelemetryEvent(
            session_id=req.session_id,
            turn_id=req.turn_id,
            controller_decisions=decisions,
            controller_decision=decisions[-1] if decisions else None,
            refinement_type=refinement_type,
            retrieval_required=True,
            retrieval_events=retrieval_events,
            sub_queries=sub_query_texts,
            answer=answer_text,
            citations=[],
            uncertainty="Information not available in provided documents",
            grounding_score=1.0,
            answer_version=session.answer_version,
            latencies_ms=LatenciesMs(
                retrieval=round(retrieval_latency, 2),
                decompose=round(decompose_latency, 2),
                refinement=round(refinement_latency, 2),
                end_to_end=round((time.time() - start_time) * 1000, 2),
            ),
        )
        emit(telemetry)
        return TurnResponse(answer=answer_text, telemetry=telemetry)

    # ── Build Context Blocks for Synthesis ─────────────────────────────
    context_blocks = []
    retrieved_tags = []
    for mc in merged_chunks:
        retrieved_tags.append(mc.tag)
        source_info = f"  (relevant to: {', '.join(mc.source_sub_queries)})" if mc.source_sub_queries else ""
        context_blocks.append(f"[{mc.tag}]{source_info}\n{mc.text}")
    context_str = "\n\n".join(context_blocks)

    # Allowed tags for validation: for LATE_DETAIL, union prior citations (C7)
    if refinement_type == "LATE_DETAIL":
        allowed_tags = list(set(session.current_citations) | set(retrieved_tags))
    else:
        allowed_tags = retrieved_tags

    # ── Build Synthesis Prompt ─────────────────────────────────────────
    grounding_rules = """
STRICT GROUNDING RULES:
1. Every factual claim MUST be cited with [Doc_XX §Y] tags from the context.
2. Do NOT invent or fabricate any Doc IDs or Section numbers.
3. If the context does not contain information to answer a question or sub-question, explicitly state:
   "This information is not available in the provided documents."
4. Never answer from your own knowledge. Only use the provided context.
"""
    if refinement_type == "LATE_DETAIL":
        prompt = f"""You are a helpful assistant answering based ONLY on the provided context.
The user is adding a new constraint to their prior question. Refine your previous answer without starting over.

Previous Answer:
{session.current_answer}

New User Constraint:
{req.utterance}
(Detected constraint: {refinement_result.get('constraint', req.utterance)})

Context:
{context_str}

{grounding_rules}
Refine the previous answer, incorporating the new constraint while preserving valid parts and citation tags.
"""
    elif len(sub_query_texts) > 1:
        sub_list = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_query_texts))
        prompt = f"""You are a helpful assistant answering based ONLY on the provided context.
The user's question contains multiple sub-questions:
{sub_list}

Context:
{context_str}

User Query:
{req.utterance}

{grounding_rules}
Answer each sub-question in order, citing factual claims with [Doc_XX §Y] tags.
"""
    else:
        prompt = f"""You are a helpful assistant answering based ONLY on the provided context.
Context:
{context_str}

User Query:
{req.utterance}

{grounding_rules}
Provide a clear, well-cited answer.
"""

    # ── LLM Synthesis (Async wrapped, H1) ──────────────────────────────
    llm_start = time.time()
    try:
        response = await asyncio.to_thread(_call_gemini_sync, prompt)
        answer_text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        token_cost = TokenCost(
            input=in_tok,
            output=out_tok,
            usd_estimate=round(in_tok * 0.075 / 1e6 + out_tok * 0.30 / 1e6, 6),
            synthesis_input_tokens=in_tok,
            synthesis_output_tokens=out_tok,
        )
    except Exception:
        # Never leak provider internals / keys to API callers.
        raise HTTPException(status_code=502, detail="LLM Synthesis error: upstream provider unavailable")

    ttft = (time.time() - llm_start) * 1000

    # ── Grounding Validation & Enforcement (C5, C6) ────────────────────
    grounding_start = time.time()
    report = validate(answer_text, allowed_tags)

    if not report.ok:
        # Regenerate ONCE with the violation explicitly named
        retry_prompt = (
            prompt +
            f"\n\nCRITICAL FIX: Do not cite {report.fabricated}. "
            f"Cite ONLY tags present in the context: {allowed_tags}. "
            "Ensure all factual claims carry citation tags."
        )
        try:
            retry_res = await asyncio.to_thread(_call_gemini_sync, retry_prompt)
            answer_text = (retry_res.text or "").strip()
            report = validate(answer_text, allowed_tags)
        except Exception:
            pass

    if not report.ok:
        # Still not grounded -> abstain rather than ship ungrounded claims (C6)
        answer_text = "I cannot answer that reliably based on the provided documents."
        report = validate(answer_text, [])

    grounding_latency = (time.time() - grounding_start) * 1000

    # Citations in telemetry are what the answer cited, not all retrieved tags
    final_citations = report.cited
    uncertainty_text = "Information not available in provided documents" if report.abstained or not report.ok else ""

    # ── Session Commit (C4, C7) ────────────────────────────────────────
    session.commit(
        kind=refinement_type,
        utterance=req.utterance,
        answer=answer_text,
        cited_tags=final_citations,
        effective_query=effective_query,
        sub_queries=sub_query_texts,
    )

    total_latency = (time.time() - start_time) * 1000

    # ── Telemetry Event (H2) ───────────────────────────────────────────
    telemetry = TelemetryEvent(
        session_id=req.session_id,
        turn_id=req.turn_id,
        controller_decisions=decisions,
        controller_decision=decisions[-1] if decisions else None,
        refinement_type=refinement_type,
        retrieval_required=True,
        retrieval_events=retrieval_events,
        sub_queries=sub_query_texts,
        answer=answer_text,
        citations=final_citations,
        uncertainty=uncertainty_text,
        grounding_score=report.score,
        grounding_report=report.to_dict(),
        answer_version=session.answer_version,
        latencies_ms=LatenciesMs(
            retrieval=round(retrieval_latency, 2),
            decompose=round(decompose_latency, 2),
            refinement=round(refinement_latency, 2),
            grounding=round(grounding_latency, 2),
            time_to_first_token=round(ttft, 2),
            end_to_end=round(total_latency, 2),
            controller=round(refinement_latency + decompose_latency, 2),
            synthesis=round(ttft, 2),
            retrieval_pipeline=round(retrieval_latency, 2),
        ),
        token_cost=token_cost,
    )
    emit(telemetry)
    return TurnResponse(answer=answer_text, telemetry=telemetry)
