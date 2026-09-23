import os
import sys
import asyncio
import time
import threading
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
from session.store import get_or_create_session, get_session_lock
from retrieval.hybrid_search import retrieve_and_rerank
from telemetry.sink import emit

import html
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

if hasattr(asyncio, "timeout"):
    _timeout = asyncio.timeout  # Python 3.11+
else:  # Python 3.10 fallback used by the local dev interpreter
    @asynccontextmanager
    async def _timeout(delay: float | None) -> AsyncIterator[None]:
        try:
            yield
        except asyncio.CancelledError:
            raise asyncio.TimeoutError from None

load_dotenv()

app = FastAPI(title="Streaming Live RAG - Live Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if os.getenv("CORS_ALLOW_ORIGINS")
    else ["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

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
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_RATE_LIMIT_TTL_S = 120.0  # forget idle sessions so the map cannot grow forever


def _sanitize_for_prompt(text: str, max_len: int = 2000) -> str:
    """Strip prompt-injection framing from user/session text before LLM use."""
    cleaned = (text or "")[:max_len]
    # Neutralize common instruction-override markers; keep the factual content.
    for marker in (
        "system:",
        "assistant:",
        "ignore previous",
        "ignore all previous",
        "disregard previous",
        "do not cite",
        "do not follow",
    ):
        cleaned = cleaned.replace(marker, " ")
    return html.unescape(cleaned).strip()


def _check_rate_limit(session_id: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        # Opportunistic eviction: drop idle entries so a session-id rotation
        # attack cannot grow this dict without bound.
        for sid in [s for s, ts in _session_request_times.items() if not ts or now - ts[-1] > _RATE_LIMIT_TTL_S]:
            _session_request_times.pop(sid, None)
        timestamps = _session_request_times.setdefault(session_id, [])
        valid = [t for t in timestamps if now - t < 60.0]
        if len(valid) >= RATE_LIMIT_PER_MINUTE:
            _session_request_times[session_id] = valid
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
            err = f"{type(e).__name__}: {str(e)}"
            is_transient = any(
                c in err for c in ("429", "500", "502", "503", "504", "ResourceExhausted", "Timeout", "timeout", "Unavailable", "DeadlineExceeded")
            )
            if attempt == max_retries - 1 or not is_transient:
                raise
            time.sleep(2 ** attempt + (os.urandom(1)[0] / 255.0))


def _conversational_reply(utterance: str) -> str:
    """Generates a brief polite response for chit-chat turns without facts or citations."""
    safe_utterance = _sanitize_for_prompt(utterance, max_len=500)
    try:
        prompt = (
            "You are a helpful, courteous assistant. Respond briefly and politely to this greeting or remark. "
            "Do NOT reference any documents, citations, or external facts:\n\n"
            f"User: {safe_utterance}"
        )
        res = _call_gemini_sync(prompt)
        return (res.text or "").strip() or "Hello! How can I assist you with the documents today?"
    except Exception as e:
        logger.warning("conversational reply failed: %s", e)
        return "Hello! How can I assist you with the documents today?"


def _reformat_reply(instruction: str, base_answer: str) -> str:
    """Reformats an existing substantive answer per presentation instruction without new facts."""
    safe_instruction = _sanitize_for_prompt(instruction, max_len=500)
    safe_base = _sanitize_for_prompt(base_answer, max_len=4000)
    try:
        prompt = f"""You are a helpful assistant. Reformat the following previous answer according to the user instruction:
User Instruction: {safe_instruction}

Previous Answer:
{safe_base}

STRICT RULES:
1. Preserve all factual claims and citation tags ([Doc_XX §Y]) exactly as they appear in the previous answer.
2. Do NOT add new factual claims or fabricate new citation tags.
3. Follow the formatting requested (e.g. bullet points, concise summary, simpler wording).
"""
        res = _call_gemini_sync(prompt)
        text = (res.text or "").strip()
        return text or base_answer
    except Exception as e:
        logger.warning("reformat reply failed: %s", e)
        return base_answer


# ---------------------------------------------------------------------------
# Main Turn Endpoint
# ---------------------------------------------------------------------------

@app.post("/turn", response_model=TurnResponse)
async def handle_turn(req: TurnRequest):
    if not _check_rate_limit(req.session_id):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded for this session (max {RATE_LIMIT_PER_MINUTE} requests/minute).")
    start_time = time.time()
    # Snapshot session state under lock, then release before awaiting LLMs.
    # The commit path re-acquires the same lock, so concurrent turns for one
    # session_id serialize on commit instead of racing turns/versions.
    session_lock = get_session_lock(req.session_id)
    with session_lock:
        session = get_or_create_session(req.session_id)
        has_history = len(session.turns) > 0
        history_context = session.get_history_context(max_turns=5)
        current_answer_snapshot = session.current_answer
        current_query_snapshot = session.current_query

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
            conversation_history=history_context,
            previous_answer=current_answer_snapshot,
        )
    refinement_type = refinement_result.get("type", "NEW_TOPIC")
    refinement_latency = (time.time() - refinement_start) * 1000
    if refinement_result.get("degraded"):
        logger.warning("refinement classifier degraded: %s", refinement_result.get("reason"))

    def _commit_locked(kind: str, utterance: str, answer: str, cited: list[str], effective_query: str = "", sub_queries: list[str] | None = None):
        with session_lock:
            live = get_or_create_session(req.session_id)
            record = live.commit(kind, utterance, answer, cited, effective_query=effective_query, sub_queries=sub_queries)
            return live.answer_version, record.turn_id

    # ── PRESENTATION_ONLY: Reformat or conversational opener (C4) ──────
    if refinement_type == "PRESENTATION_ONLY":
        if current_answer_snapshot:
            # Reformat existing substantive answer
            with session_lock:
                allowed_cites = list(get_or_create_session(req.session_id).current_citations)
            answer_text = await asyncio.to_thread(_reformat_reply, req.utterance, current_answer_snapshot)
            # Verify no citations were fabricated during reformatting
            check_report = validate(answer_text, allowed_cites)
            if check_report.fabricated:
                answer_text = current_answer_snapshot
                check_report = validate(answer_text, allowed_cites)
            citations = check_report.cited
        else:
            # Conversational opener / chit-chat with no prior answer
            answer_text = await asyncio.to_thread(_conversational_reply, req.utterance)
            citations = []

        answer_version, server_turn_id = _commit_locked("PRESENTATION_ONLY", req.utterance, answer_text, citations)

        telemetry = TelemetryEvent(
            session_id=req.session_id,
            turn_id=server_turn_id,
            refinement_type="PRESENTATION_ONLY",
            retrieval_required=False,
            retrieval_skip_reason=refinement_result.get("reason", "presentation_only"),
            retrieval_events=[],
            sub_queries=[],
            answer=answer_text,
            citations=citations,
            answer_version=answer_version,
            degraded=bool(refinement_result.get("degraded")),
            latencies_ms=LatenciesMs(
                refinement=round(refinement_latency, 2),
                controller=round(refinement_latency, 2),
                end_to_end=round((time.time() - start_time) * 1000, 2),
            ),
        )
        try:
            emit(telemetry)
        except Exception as e:
            logger.warning("telemetry emit failed: %s", e)
        return TurnResponse(answer=answer_text, telemetry=telemetry)

    # ── Phase 3: Two-Stage Streaming Controller (C2 / C3 / H4) ──────────
    chunks = list(simulate_stream(req.utterance, words_per_chunk=2, ms_per_chunk=300))
    t_end = chunks[-1].t_offset_s if chunks else 0.0
    decisions: list[ControllerDecision] = []
    retrieval_events: list[RetrievalEvent] = []
    degraded = bool(refinement_result.get("degraded"))
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
        if d.get("degraded"):
            degraded = True
        decision_record = ControllerDecision(
            trigger=d["trigger"],
            timestamp_s=chunk.t_offset_s,
            reason=d.get("reason", ""),
        )
        decisions.append(decision_record)

        if d["trigger"] == "no_retrieval_needed":
            # Early exit: Chit-chat detected (C2)
            answer_text = await asyncio.to_thread(_conversational_reply, req.utterance)
            answer_version, server_turn_id = _commit_locked("PRESENTATION_ONLY", req.utterance, answer_text, [])
            if provisional is not None:
                provisional[1].cancel()
            telemetry = TelemetryEvent(
                session_id=req.session_id,
                turn_id=server_turn_id,
                controller_decisions=decisions,
                controller_decision=decision_record,
                refinement_type="PRESENTATION_ONLY",
                retrieval_required=False,
                retrieval_skip_reason=d.get("reason", "no_retrieval_needed"),
                retrieval_events=[],
                sub_queries=[],
                answer=answer_text,
                citations=[],
                answer_version=answer_version,
                degraded=degraded,
                latencies_ms=LatenciesMs(
                    controller=round((time.time() - start_time) * 1000, 2),
                    end_to_end=round((time.time() - start_time) * 1000, 2),
                ),
            )
            try:
                emit(telemetry)
            except Exception as e:
                logger.warning("telemetry emit failed: %s", e)
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
        constraint = _sanitize_for_prompt(refinement_result.get("constraint") or req.utterance, max_len=500)
        effective_query = f"{current_query_snapshot}. Additional constraint: {constraint}"
    else:
        effective_query = req.utterance

    sub_query_dicts = await asyncio.to_thread(decompose_query, effective_query)
    decompose_latency = (time.time() - decompose_start) * 1000
    if any(isinstance(sq, dict) and sq.get("degraded") for sq in sub_query_dicts):
        degraded = True

    sub_query_texts = [
        sq["sub_query"] for sq in sub_query_dicts if isinstance(sq, dict) and sq.get("sub_query")
    ]
    if not sub_query_texts:
        # Only fall back to a full retrieval query for substantive turns.
        # PRESENTATION_ONLY is handled above, so an empty decomposer result
        # here means chit-chat slipped through — answer conversationally.
        logger.warning("decomposer returned no sub-queries; using conversational fallback")
        answer_text = await asyncio.to_thread(_conversational_reply, req.utterance)
        answer_version, server_turn_id = _commit_locked("PRESENTATION_ONLY", req.utterance, answer_text, [])
        if provisional is not None:
            provisional[1].cancel()
        telemetry = TelemetryEvent(
            session_id=req.session_id,
            turn_id=server_turn_id,
            controller_decisions=decisions,
            controller_decision=decisions[-1] if decisions else None,
            refinement_type="PRESENTATION_ONLY",
            retrieval_required=False,
            retrieval_skip_reason="decomposer returned no sub-queries",
            retrieval_events=[],
            sub_queries=[],
            answer=answer_text,
            citations=[],
            answer_version=answer_version,
            degraded=degraded,
            latencies_ms=LatenciesMs(
                decompose=round(decompose_latency, 2),
                refinement=round(refinement_latency, 2),
                controller=round(refinement_latency + decompose_latency, 2),
                end_to_end=round((time.time() - start_time) * 1000, 2),
            ),
        )
        try:
            emit(telemetry)
        except Exception as e:
            logger.warning("telemetry emit failed: %s", e)
        return TurnResponse(answer=answer_text, telemetry=telemetry)

    # Calculate delta: Sub-queries not already covered by provisional retrieval
    todo = [
        q for q in sub_query_texts
        if not _same_intent(q, provisional[0] if provisional else None)
    ]

    retrieval_start = time.time()
    delta_hits: list = []
    prov_hits: list = []
    retrieval_error: str | None = None
    try:
        async with _timeout(30):
            delta_tasks = [asyncio.to_thread(retrieve_and_rerank, q, q, 5) for q in todo]
            if delta_tasks:
                delta_results = await asyncio.gather(*delta_tasks, return_exceptions=True)
                for res in delta_results:
                    if isinstance(res, BaseException):
                        retrieval_error = str(res)
                        delta_hits.append([])
                    else:
                        delta_hits.append(res)
            if provisional is not None:
                try:
                    prov_hits = await provisional[1]
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    retrieval_error = str(e)
                    prov_hits = []
    except (asyncio.TimeoutError, asyncio.CancelledError):
        if provisional is not None and not provisional[1].done():
            provisional[1].cancel()
        raise HTTPException(status_code=504, detail="Retrieval timed out; please retry.")
    if retrieval_error:
        logger.warning("retrieval partially failed: %s", retrieval_error)
        degraded = True
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
    # Cross-encoder scores are corpus/model-specific; keep the floor opt-in via
    # env so a stale magic number cannot silently abstain on good evidence.
    relevance_floor = float(os.getenv("RELEVANCE_FLOOR", "-999"))
    best_score = max((c.best_score for c in merged_chunks), default=-999.0)
    if not merged_chunks or best_score < relevance_floor:
        # Out-of-corpus / unanswerable request
        answer_text = "This information is not available in the provided documents."
        answer_version, server_turn_id = _commit_locked(refinement_type, req.utterance, answer_text, [], effective_query=effective_query)
        telemetry = TelemetryEvent(
            session_id=req.session_id,
            turn_id=server_turn_id,
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
            answer_version=answer_version,
            degraded=degraded,
            latencies_ms=LatenciesMs(
                retrieval=round(retrieval_latency, 2),
                decompose=round(decompose_latency, 2),
                refinement=round(refinement_latency, 2),
                controller=round(refinement_latency + decompose_latency, 2),
                end_to_end=round((time.time() - start_time) * 1000, 2),
            ),
        )
        try:
            emit(telemetry)
        except Exception as e:
            logger.warning("telemetry emit failed: %s", e)
        return TurnResponse(answer=answer_text, telemetry=telemetry)

    # ── Build Context Blocks for Synthesis ─────────────────────────────
    context_blocks = []
    retrieved_tags = []
    for mc in merged_chunks:
        retrieved_tags.append(mc.tag)
        source_info = f"  (relevant to: {', '.join(mc.source_sub_queries)})" if mc.source_sub_queries else ""
        context_blocks.append(f"[{mc.tag}]{source_info}\n{mc.text}")
    context_str = "\n\n".join(context_blocks)

    # Allowed tags for validation are ONLY what was retrieved for THIS turn.
    # Stale prior-turn citations are never valid evidence for new claims;
    # session union still happens on commit for lineage (store.py), but the
    # LLM must cite current context.
    with session_lock:
        prior_answer_snapshot = get_or_create_session(req.session_id).current_answer
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
        safe_constraint = _sanitize_for_prompt(refinement_result.get('constraint', req.utterance), max_len=500)
        safe_utterance = _sanitize_for_prompt(req.utterance, max_len=1000)
        safe_prior = _sanitize_for_prompt(prior_answer_snapshot, max_len=4000)
        prompt = f"""You are a helpful assistant answering based ONLY on the provided context.
The user is adding a new constraint to their prior question. Refine your previous answer without starting over.

Previous Answer:
{safe_prior}

New User Constraint:
{safe_utterance}
(Detected constraint: {safe_constraint})

Context:
{context_str}

{grounding_rules}
Refine the previous answer, incorporating the new constraint while preserving valid parts and citation tags.
"""
    elif len(sub_query_texts) > 1:
        sub_list = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_query_texts))
        safe_utterance = _sanitize_for_prompt(req.utterance, max_len=1000)
        prompt = f"""You are a helpful assistant answering based ONLY on the provided context.
The user's question contains multiple sub-questions:
{sub_list}

Context:
{context_str}

User Query:
{safe_utterance}

{grounding_rules}
Answer each sub-question in order, citing factual claims with [Doc_XX §Y] tags.
"""
    else:
        safe_utterance = _sanitize_for_prompt(req.utterance, max_len=1000)
        prompt = f"""You are a helpful assistant answering based ONLY on the provided context.
Context:
{context_str}

User Query:
{safe_utterance}

{grounding_rules}
Provide a clear, well-cited answer.
"""

    # ── LLM Synthesis (Async wrapped, H1) ──────────────────────────────
    llm_start = time.time()
    try:
        async with _timeout(60):
            response = await asyncio.to_thread(_call_gemini_sync, prompt)
        answer_text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        token_cost = TokenCost(
            input=in_tok,
            output=out_tok,
            usd_estimate=round(in_tok * float(os.getenv("SYNTHESIS_PRICE_PER_M_INPUT", "0.075")) / 1e6 + out_tok * float(os.getenv("SYNTHESIS_PRICE_PER_M_OUTPUT", "0.30")) / 1e6, 6),
            synthesis_input_tokens=in_tok,
            synthesis_output_tokens=out_tok,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        raise HTTPException(status_code=504, detail="LLM Synthesis timed out; please retry.")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("LLM synthesis failed: %s", e)
        # Never leak provider internals / keys to API callers.
        raise HTTPException(status_code=502, detail="LLM Synthesis error: upstream provider unavailable")

    ttft = (time.time() - llm_start) * 1000

    # ── Grounding Validation & Enforcement (C5, C6) ────────────────────
    grounding_start = time.time()
    report = validate(answer_text, allowed_tags)

    if not report.ok:
        # Regenerate ONCE with the violation explicitly named
        fabricated_list = ", ".join(report.fabricated) if report.fabricated else "uncited claims"
        retry_prompt = (
            prompt +
            f"\n\nCRITICAL FIX: Your previous answer contained {fabricated_list}, "
            f"which are not valid evidence. Cite ONLY these exact tags from the context: {', '.join(allowed_tags)}. "
            "Ensure all factual claims carry citation tags."
        )
        try:
            async with _timeout(60):
                retry_res = await asyncio.to_thread(_call_gemini_sync, retry_prompt)
            answer_text = (retry_res.text or "").strip()
            report = validate(answer_text, allowed_tags)
        except Exception as e:
            logger.warning("grounding retry failed: %s", e)

    if not report.ok:
        # Still not grounded -> abstain rather than ship ungrounded claims (C6)
        answer_text = "I cannot answer that reliably based on the provided documents."
        report = validate(answer_text, [])

    grounding_latency = (time.time() - grounding_start) * 1000

    # Citations in telemetry are what the answer cited, not all retrieved tags
    final_citations = report.cited
    uncertainty_text = "Information not available in provided documents" if report.abstained or not report.ok else ""

    # ── Session Commit (C4, C7) ────────────────────────────────────────
    answer_version, server_turn_id = _commit_locked(
        kind=refinement_type,
        utterance=req.utterance,
        answer=answer_text,
        cited=final_citations,
        effective_query=effective_query,
        sub_queries=sub_query_texts,
    )

    total_latency = (time.time() - start_time) * 1000

    # ── Telemetry Event (H2) ───────────────────────────────────────────
    telemetry = TelemetryEvent(
        session_id=req.session_id,
        turn_id=server_turn_id,
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
        answer_version=answer_version,
        degraded=degraded,
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
    try:
        emit(telemetry)
    except Exception as e:
        logger.warning("telemetry emit failed: %s", e)
    return TurnResponse(answer=answer_text, telemetry=telemetry)
