import os
import sys
import asyncio
import uuid
import time
import threading
import hmac
import re
import ipaddress
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telemetry.schema import TelemetryEvent
from dotenv import load_dotenv

from streaming.live_stream import play_utterance, LiveQueueSource
from streaming.engine import run_live_turn
from retrieval.ingest import (
    ingest_file_or_text,
    get_corpus_summary,
    clear_corpus,
)
from llm_config import (
    get_llm_config,
    update_llm_config,
    clear_llm_key,
    test_llm_connection,
)

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

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB max upload ceiling to prevent memory exhaustion / zip-bombs

app = FastAPI(title="Streaming Live RAG - Live Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if os.getenv("CORS_ALLOW_ORIGINS")
    else ["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Admin-Token"],
)


def _is_loopback_or_docker(client_host: str) -> bool:
    """
    Checks if client_host is loopback, Docker bridge gateway, or internal container network.
    When running containerized in Docker, requests originating from the host arrive
    via the Docker bridge network gateway (e.g. 172.18.0.1 or 192.168.65.1).
    """
    if not client_host or client_host in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(client_host)
        if ip.is_loopback:
            return True
        # Allow Docker bridge subnets and private host gateways (172.16.0.0/12, 192.168.0.0/16, 10.0.0.0/8)
        if (
            ip in ipaddress.ip_network("172.16.0.0/12")
            or ip in ipaddress.ip_network("192.168.0.0/16")
            or ip in ipaddress.ip_network("10.0.0.0/8")
        ):
            return True
    except ValueError:
        pass
    return False


def verify_admin_access(request: Request) -> bool:
    """
    Enforces access control on administrative and mutation endpoints (/config/*, /corpus/clear, /upload).
    - If ADMIN_TOKEN is set: requires matching token via X-Admin-Token or Authorization: Bearer.
      Verified using constant-time hmac.compare_digest to prevent timing attacks.
    - If ADMIN_TOKEN is unset: strictly serves loopback and local Docker container environments.
      External public callers receive HTTP 403.
    """
    client_host = request.client.host if request.client else ""
    is_loopback = _is_loopback_or_docker(client_host)

    if not ADMIN_TOKEN:
        if is_loopback:
            return True
        raise HTTPException(
            status_code=403,
            detail="ADMIN_TOKEN is not configured on server; non-loopback access to admin endpoints is forbidden. Please configure ADMIN_TOKEN in the server environment."
        )

    auth_header = request.headers.get("Authorization", "")
    admin_header = request.headers.get("X-Admin-Token", "")
    token = ""
    if admin_header:
        token = admin_header.strip()
    elif auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()

    if not token or not hmac.compare_digest(token.encode("utf-8"), ADMIN_TOKEN.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing admin authentication token.")
    return True


# Concurrency & DoS guards
_turn_semaphore = asyncio.Semaphore(10)
_session_request_times: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_RATE_LIMIT_TTL_S = 120.0



IP_RATE_LIMIT_PER_MINUTE = int(os.getenv("IP_RATE_LIMIT_PER_MINUTE", "300"))


def _check_rate_limit(session_id: str, client_ip: str = "127.0.0.1") -> bool:
    """
    Dual-bucket rate limiter:
    1. Per-(IP, session) bucket: RATE_LIMIT_PER_MINUTE (default 60/min).
    2. Aggregate per-IP bucket: IP_RATE_LIMIT_PER_MINUTE (default 300/min).
       Note on shared-NAT/corporate proxies: 300 req/min applies to the shared public egress IP.
    Sweeps both key types on TTL expiration to prevent memory growth.
    """
    pair_key = f"{client_ip}:{session_id}"
    ip_key = f"ip:{client_ip}"
    now = time.time()
    with _rate_limit_lock:
        # TTL eviction sweep for both pair keys and aggregate ip: keys
        expired_keys = [k for k, ts in _session_request_times.items() if not ts or (now - ts[-1] > _RATE_LIMIT_TTL_S)]
        for k in expired_keys:
            _session_request_times.pop(k, None)

        # 1. Check aggregate per-IP bucket
        ip_timestamps = _session_request_times.setdefault(ip_key, [])
        valid_ip = [t for t in ip_timestamps if now - t < 60.0]
        if len(valid_ip) >= IP_RATE_LIMIT_PER_MINUTE:
            _session_request_times[ip_key] = valid_ip
            return False

        # 2. Check per-(IP, session) bucket
        pair_timestamps = _session_request_times.get(pair_key)
        if pair_timestamps is None:
            pair_timestamps = _session_request_times.get(session_id)
        if pair_timestamps is None:
            pair_timestamps = []
            _session_request_times[pair_key] = pair_timestamps

        valid_pair = [t for t in pair_timestamps if now - t < 60.0]
        if len(valid_pair) >= RATE_LIMIT_PER_MINUTE:
            _session_request_times[pair_key] = valid_pair
            return False

        # Append timestamp to both buckets
        valid_ip.append(now)
        _session_request_times[ip_key] = valid_ip
        valid_pair.append(now)
        _session_request_times[pair_key] = valid_pair
        return True


@app.on_event("startup")
def _auto_seed_corpus_on_startup():
    """Ensures indexed corpus strictly contains ONLY Doc 1 and Doc 2 whenever server starts."""
    try:
        from retrieval.ingest import parse_corpus, ingest_sections, get_corpus_summary, clear_corpus
        summary = get_corpus_summary()
        docs = summary.get("documents", [])
        doc_ids = {d.get("doc_id") for d in docs}
        # Whenever start: keep strictly Doc 1 and Doc 2 on the indexed corpus
        if doc_ids != {"Doc_01", "Doc_02"} or summary.get("total_chunks", 0) != 4:
            corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dev_corpus"))
            if os.path.exists(corpus_dir):
                chunks = parse_corpus(corpus_dir, allowed_docs={"Doc_01", "Doc_02"})
                if chunks:
                    clear_corpus()
                    ingest_sections(chunks, reset=True)
                    logger.info("Startup: indexed strictly Doc 1 and Doc 2 (%d chunks) into corpus.", len(chunks))
    except Exception as e:
        logger.warning("Auto-seed on startup failed: %s", e)


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


@app.get("/system/hardware")
def get_hardware_acceleration():
    """Returns detected GPU hardware, execution provider, and acceleration status."""
    try:
        from retrieval.hybrid_search import get_gpu_info
        return get_gpu_info()
    except Exception as e:
        return {"gpu_available": False, "device_name": "CPU", "provider": "CPUExecutionProvider", "error": str(e)}


@app.get("/")
@app.get("/demo")
def demo_ui():
    """Serves the interactive live streaming demo dashboard."""
    static_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "index.html"))
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Streaming Live RAG API is running. Access /turn for API queries."}


@app.get("/corpus")
@app.get("/documents/summary")
def get_corpus():
    """Returns currently indexed documents, total chunks, and citation tags."""
    return get_corpus_summary()


@app.post("/corpus/clear")
def clear_all_corpus(_authorized: bool = Depends(verify_admin_access)):
    """Resets the vector collection to empty (Admin authorized)."""
    ok = clear_corpus()
    return {"status": "cleared" if ok else "failed"}


@app.post("/corpus/reset")
@app.post("/corpus/reseed")
def reset_to_dev_corpus(_authorized: bool = Depends(verify_admin_access)):
    """Resets the vector collection and re-indexes strictly Doc 1 and Doc 2 (Admin authorized)."""
    from retrieval.ingest import parse_corpus, ingest_sections, clear_corpus
    clear_corpus()
    corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dev_corpus"))
    chunks = parse_corpus(corpus_dir, allowed_docs={"Doc_01", "Doc_02"})
    res = ingest_sections(chunks, reset=True) if chunks else {"chunks_indexed": 0}
    return {
        "status": "reseeded",
        "chunks_indexed": res.get("chunks_indexed", len(chunks)),
        "summary": get_corpus_summary(),
    }


@app.post("/upload")
@app.post("/corpus/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(None),
    text: str = Form(None),
    filename: str = Form(None),
    reset: bool = Form(False),
    _authorized: bool = Depends(verify_admin_access),
):
    """
    Uploads and indexes arbitrary document files or raw text into the live Qdrant vector database.
    Guaranteed serialized, content-hash deduplicated, and non-blocking to the event loop.
    """
    if file is None and not (text or "").strip():
        raise HTTPException(status_code=400, detail="Either a file upload or text content must be provided.")

    # Guard against memory-exhaustion / zip-bomb DoS
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Upload exceeds maximum permitted size of {MAX_UPLOAD_BYTES // (1024*1024)}MB.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

    if file is not None:
        fname = filename or file.filename or "uploaded_document.txt"
        try:
            content_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File exceeds maximum permitted size of {MAX_UPLOAD_BYTES // (1024*1024)}MB.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")
    else:
        fname = filename or "custom_notes.txt"
        content_bytes = text.encode("utf-8")
        if len(content_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Text exceeds maximum permitted size of {MAX_UPLOAD_BYTES // (1024*1024)}MB.")

    if not content_bytes.strip():
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")

    t0 = time.time()
    try:
        res = await asyncio.to_thread(
            ingest_file_or_text,
            filename=fname,
            content_bytes=content_bytes,
            text_override=None,
            reset=reset
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error("Indexing failed for %s: %s", fname, e)
        raise HTTPException(status_code=500, detail=f"Vector indexing failed: {e}")

    indexing_ms = round((time.time() - t0) * 1000, 2)
    return {
        "status": res.get("status", "success"),
        "message": res.get("message", "Document indexed successfully."),
        "filename": fname,
        "doc_id": res.get("doc_id", ""),
        "sections_indexed": res.get("chunks_indexed", 0),
        "tags": res.get("tags", []),
        "indexing_time_ms": indexing_ms,
    }


class LLMConfigUpdate(BaseModel):
    fast_provider: str | None = None
    fast_api_key: str | None = None
    fast_base_url: str | None = None
    fast_model: str | None = None
    synthesis_provider: str | None = None
    synthesis_api_key: str | None = None
    synthesis_base_url: str | None = None
    synthesis_model: str | None = None


@app.get("/config/llm")
@app.get("/admin/llm/config")
def get_current_llm_config(_authorized: bool = Depends(verify_admin_access)):
    """Returns currently active LLM providers and models adhering to write-only keys architecture."""
    return get_llm_config()


@app.post("/config/llm")
@app.post("/admin/llm/config")
def update_current_llm_config(cfg: LLMConfigUpdate, _authorized: bool = Depends(verify_admin_access)):
    """Updates runtime LLM provider settings (BYOK) without restarting the server."""
    updates = cfg.model_dump(exclude_unset=True)
    try:
        return update_llm_config(updates)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.delete("/config/llm/key")
@app.delete("/admin/llm/key")
def delete_llm_key(slot: str, _authorized: bool = Depends(verify_admin_access)):
    """Explicitly removes a configured key for slot ('fast' or 'synthesis')."""
    if slot not in ("fast", "synthesis"):
        raise HTTPException(status_code=400, detail="Invalid slot: must be 'fast' or 'synthesis'.")
    ok = clear_llm_key(slot)
    return {"status": "cleared" if ok else "not_found", "slot": slot}


@app.post("/config/llm/test")
@app.post("/admin/llm/test")
async def test_providers(target: str = "both", _authorized: bool = Depends(verify_admin_access)):
    """Pings configured providers to test credentials and measure latency."""
    return await asyncio.to_thread(test_llm_connection, target=target)


class SessionResetRequest(BaseModel):
    session_id: str | None = None


@app.post("/session/reset")
@app.post("/session/{session_id}/reset")
@app.delete("/session/{session_id}")
def reset_session_endpoint(session_id: str | None = None, req: SessionResetRequest | None = None):
    """
    Clears conversation memory, turn history, and fast query cache.
    - If session_id is provided, deletes that specific session lineage.
    - If no session_id is provided, clears all ephemeral sessions.
    """
    from session.store import delete_session, reset_store
    from streaming.engine import clear_query_cache
    clear_query_cache()
    target = session_id or (req.session_id if req else None)
    if target:
        deleted = delete_session(target)
        return {"status": "cleared" if deleted else "not_found", "session_id": target, "existed": deleted}
    reset_store()
    return {"status": "cleared", "all": True}


class TurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$")
    turn_id: int = Field(..., ge=1, le=10000)
    utterance: str = Field(..., min_length=1, max_length=1000)
    live_mode: bool = Field(default=True)


class TurnResponse(BaseModel):
    answer: str
    telemetry: TelemetryEvent
    thoughts: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Turn pipeline — the implementation lives in streaming/engine.py (single
# source of truth). _same_intent is re-exported here for back-compat.
# ---------------------------------------------------------------------------
from streaming.engine import _same_intent  # noqa: F401,E402


# ---------------------------------------------------------------------------
# Turn endpoints — one live engine, two transports
# ---------------------------------------------------------------------------

@app.post("/turn", response_model=TurnResponse)
async def handle_turn(req: TurnRequest, request: Request):
    """
    Replay ``req.utterance`` as a live stream or execute standard sequential RAG.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(req.session_id, client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for this session/IP (max {RATE_LIMIT_PER_MINUTE} requests/minute).",
        )

    async with _turn_semaphore:
        try:
            replayed_thoughts: list[dict] = []

            async def _collect_thought(ev: dict):
                if ev.get("type") == "thought":
                    replayed_thoughts.append(ev)

            if req.live_mode:
                source = play_utterance(req.utterance, words_per_chunk=2, ms_per_chunk=300)
            else:
                source = play_utterance(req.utterance, words_per_chunk=len(req.utterance.split()) or 1, ms_per_chunk=0)
            result = await run_live_turn(req.session_id, source, emit_event=_collect_thought, live_mode=req.live_mode)
        except (asyncio.TimeoutError, TimeoutError):
            raise HTTPException(status_code=504, detail="Turn timed out; please retry.")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("/turn failed: %s", e)
            # Never leak provider internals / keys to API callers.
            raise HTTPException(status_code=502, detail="Turn failed: upstream provider unavailable")
        return TurnResponse(answer=result.answer, telemetry=result.telemetry, thoughts=replayed_thoughts)


class CompareRequest(BaseModel):
    utterance: str = Field(..., min_length=3, max_length=1000)
    words_per_chunk: int = Field(default=2, ge=1, le=10)
    ms_per_chunk: int = Field(default=250, ge=0, le=1000)


class CompareResponse(BaseModel):
    utterance: str
    live: TurnResponse
    normal: TurnResponse
    speech_duration_s: float
    live_wait_s: float
    normal_wait_s: float
    head_start_s: float
    speedup_factor: float


@app.post("/turn/compare", response_model=CompareResponse)
async def handle_compare(req: CompareRequest, request: Request):
    """
    Executes an honest side-by-side run of Streaming Live RAG vs. Normal Sequential RAG
    on the exact same utterance under identical system conditions.
    """
    client_ip = request.client.host if request.client else "unknown"
    session_prefix = f"cmp_{int(time.time() * 1000) % 1000000}"
    live_session = f"{session_prefix}_live"
    norm_session = f"{session_prefix}_norm"

    if not _check_rate_limit(live_session, client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
        )

    words = req.utterance.strip().split()
    total_words = len(words)
    chunk_count = (total_words + req.words_per_chunk - 1) // req.words_per_chunk
    simulated_speech_s = round((chunk_count * req.ms_per_chunk) / 1000.0, 2)

    try:
        # 1. Setup Live Turn (real streaming chunk pacing + speculative retrieval, distinct session, no cache)
        live_thoughts: list[dict] = []
        async def _collect_live(ev: dict):
            if ev.get("type") == "thought":
                live_thoughts.append(ev)

        live_source = play_utterance(req.utterance, words_per_chunk=req.words_per_chunk, ms_per_chunk=req.ms_per_chunk)

        async def _run_live():
            t0_l = time.time()
            res = await run_live_turn(live_session, live_source, emit_event=_collect_live, live_mode=True, use_cache=False)
            t_total = time.time() - t0_l
            return res, t_total

        # 2. Setup Normal Sequential Turn (distinct session, no cache)
        norm_thoughts: list[dict] = []
        async def _collect_norm(ev: dict):
            if ev.get("type") == "thought":
                norm_thoughts.append(ev)

        async def _run_norm():
            # Mirror realistic speech timeline: normal pipeline only starts after user finishes speaking
            if simulated_speech_s > 0:
                await asyncio.sleep(simulated_speech_s)
            t0_n = time.time()
            norm_source = play_utterance(req.utterance, words_per_chunk=len(words) or 1, ms_per_chunk=0)
            res = await run_live_turn(norm_session, norm_source, emit_event=_collect_norm, live_mode=False, use_cache=False)
            t_wait = time.time() - t0_n
            return res, t_wait

        # Run both pipelines simultaneously side-by-side!
        (live_res, live_total_s), (norm_res, normal_wait_s) = await asyncio.gather(_run_live(), _run_norm())

        live_end_s = live_res.telemetry.utterance_end_s or simulated_speech_s
        live_wait_s = max(0.0, round(live_total_s - live_end_s, 2))
        normal_wait_s = round(normal_wait_s, 2)

        # Calculate real head start and speedup
        prov_fired = live_res.telemetry.provisional_fired_s
        head_start_s = round(live_end_s - prov_fired, 2) if prov_fired is not None else 0.0
        speedup = round(normal_wait_s / max(0.05, live_wait_s), 1) if live_wait_s > 0 else 10.0

        return CompareResponse(
            utterance=req.utterance,
            live=TurnResponse(answer=live_res.answer, telemetry=live_res.telemetry, thoughts=live_thoughts),
            normal=TurnResponse(answer=norm_res.answer, telemetry=norm_res.telemetry, thoughts=norm_thoughts),
            speech_duration_s=simulated_speech_s,
            live_wait_s=live_wait_s,
            normal_wait_s=normal_wait_s,
            head_start_s=head_start_s,
            speedup_factor=speedup,
        )
    except Exception as e:
        logger.warning("Side-by-side compare failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Comparison execution failed: {e}")
    finally:
        from session.store import delete_session
        delete_session(live_session)
        delete_session(norm_session)


@app.websocket("/ws/compare")
@app.websocket("/ws/dual_stream")
async def ws_compare(ws: WebSocket):
    """
    Live voice comparison stream.
    Accepts speech chunks from the user's microphone in real time and routes
    the identical audio/text stream to both pipelines simultaneously:
    - Lane 1 (Streaming Live RAG): Consumes partial chunks in-flight, evaluating
      Gate G2 stability and firing speculative retrieval while speaker is talking.
    - Lane 2 (Normal Sequential RAG): Waits until speech completely ends, then
      begins sequential decomposition, vector search, and synthesis.
    Both run in completely isolated sessions with query cache strictly bypassed.
    """
    await ws.accept()
    session_prefix = f"cmp_voice_{int(time.time() * 1000) % 1000000}"
    live_session = f"{session_prefix}_live"
    norm_session = f"{session_prefix}_norm"

    live_source: LiveQueueSource = LiveQueueSource()
    live_thoughts: list[dict] = []
    norm_thoughts: list[dict] = []

    async def _send_live(ev: dict):
        try:
            if ev.get("type") == "thought":
                live_thoughts.append(ev)
            await ws.send_json({"lane": "live", **ev})
        except Exception:
            pass

    async def _send_norm(ev: dict):
        try:
            if ev.get("type") == "thought":
                norm_thoughts.append(ev)
            await ws.send_json({"lane": "normal", **ev})
        except Exception:
            pass

    live_engine_task: asyncio.Task | None = None
    t_speech_start = time.time()
    t_speech_end = time.time()
    last_text = ""

    try:
        await ws.send_json({
            "type": "session_init",
            "live_session": live_session,
            "norm_session": norm_session,
        })

        # Start Lane 1 (Streaming Live RAG) task immediately
        live_engine_task = asyncio.create_task(
            run_live_turn(live_session, live_source, emit_event=_send_live, live_mode=True, use_cache=False)
        )

        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=60.0)
            except (asyncio.TimeoutError, TimeoutError):
                await live_source.finish()
                break

            if not isinstance(msg, dict):
                continue

            mtype = msg.get("type")
            if mtype == "chunk":
                chunk_text = str(msg.get("text", "")).strip()
                if chunk_text:
                    last_text = chunk_text
                    await live_source.push_text(last_text)
                    # Notify normal lane that speaker is active
                    await ws.send_json({
                        "lane": "normal",
                        "type": "listening",
                        "text": "User speaking... Classic sequential RAG is idle, waiting for speech completion.",
                    })
            elif mtype == "end":
                if msg.get("text"):
                    last_text = str(msg.get("text")).strip()
                    await live_source.push_text(last_text)
                await live_source.finish()
                break
            elif mtype == "cancel":
                if live_engine_task and not live_engine_task.done():
                    live_engine_task.cancel()
                await live_source.finish()
                return

        t_speech_end = time.time()
        speech_duration_s = max(0.1, round(t_speech_end - t_speech_start, 2))
        await ws.send_json({
            "type": "speech_ended",
            "duration_s": speech_duration_s,
            "final_transcript": last_text,
        })

        # Speech is concluded! Now Lane 2 starts its sequential run on the exact same voice text
        words = last_text.strip().split()
        norm_source = play_utterance(last_text, words_per_chunk=len(words) or 1, ms_per_chunk=0)

        async def _run_normal():
            t0_n = time.time()
            res = await run_live_turn(norm_session, norm_source, emit_event=_send_norm, live_mode=False, use_cache=False)
            t_norm_wait = time.time() - t0_n
            return res, t_norm_wait

        norm_task = asyncio.create_task(_run_normal())

        # Lane 1 finishes (was pre-retrieving in background during speech!)
        live_res = await live_engine_task
        t_live_done = time.time()
        live_wait_s = max(0.0, round(t_live_done - t_speech_end, 2))

        await ws.send_json({
            "lane": "live",
            "type": "done",
            "answer": live_res.answer,
            "wait_s": live_wait_s,
            "telemetry": live_res.telemetry.model_dump(),
        })

        # Wait for Normal Lane to finish its post-speech sequential work
        norm_res, norm_wait_s = await norm_task
        normal_wait_s = round(norm_wait_s, 2)

        await ws.send_json({
            "lane": "normal",
            "type": "done",
            "answer": norm_res.answer,
            "wait_s": normal_wait_s,
            "telemetry": norm_res.telemetry.model_dump(),
        })

        prov_fired = live_res.telemetry.provisional_fired_s
        head_start_s = round(speech_duration_s - prov_fired, 2) if prov_fired is not None else 0.0
        speedup = round(normal_wait_s / max(0.05, live_wait_s), 1) if live_wait_s > 0 else 10.0

        await ws.send_json({
            "type": "summary",
            "utterance": last_text,
            "speech_duration_s": speech_duration_s,
            "live_wait_s": live_wait_s,
            "normal_wait_s": normal_wait_s,
            "head_start_s": head_start_s,
            "speedup_factor": speedup,
        })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("ws_compare error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if live_engine_task and not live_engine_task.done():
            live_engine_task.cancel()
        from session.store import delete_session
        delete_session(live_session)
        delete_session(norm_session)
        try:
            await ws.close()
        except Exception:
            pass


_WS_SESSION_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")
# Idle-only safety net (no message at all from the client for this long).
# This is distinct from the client's 60s hard duration cap on an active turn
# (static/index.html TURN_TIMEOUT_MS) — this timer only fires on silence.
# Lowered from 180s (3x the spec's 60s ceiling) since nothing else exercised
# it at that size.
_WS_IDLE_TIMEOUT_S = 90.0


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    A genuinely live turn over a WebSocket.

    Client -> server (JSON messages):
        {"type": "start", "session_id": "abc123"}   # first message; id is validated
        {"type": "chunk", "text": "<cumulative partial transcript>"}
        {"type": "end"}

    Server -> client (pushed in real time as the pipeline fires):
        {"type": "stream_started", "session_id", "t_s"}
        {"type": "transcript", "t_s", "partial_text"}
        {"type": "controller", "t_s", "trigger", "reason", "query"}
        {"type": "retrieval_started", "t_s", "trigger": "provisional", "query"}
        {"type": "utterance_end", "t_s", "full_text"}
        {"type": "status", "t_s", "stage": "refining|decomposing|decomposed|retrieving|synthesizing|grounding"}
        {"type": "retrieval", "t_s", "trigger", "query", "hits"}
        {"type": "answer", "t_s", "answer", "citations", "uncertainty",
         "grounding_score", "answer_version", "sub_queries"}
        {"type": "telemetry", "telemetry": {...}}
        {"type": "error", "message"}

    Each chunk is timestamped by the server on arrival with a wall clock, so
    the controller only ever sees what has actually arrived so far. If the
    client goes quiet for _WS_IDLE_TIMEOUT_S the stream is auto-finished.
    """
    await ws.accept()
    try:
        hello = await asyncio.wait_for(ws.receive_json(), timeout=15.0)
    except Exception:
        await ws.close(code=4400)
        return
    if not isinstance(hello, dict) or hello.get("type") != "start":
        try:
            await ws.send_json({"type": "error",
                                "message": 'first message must be {"type": "start", "session_id": "..."}'})
        finally:
            await ws.close(code=4400)
        return
    session_id = str(hello.get("session_id") or "")
    if not _WS_SESSION_RE.match(session_id):
        session_id = "ws_" + uuid.uuid4().hex[:12]
    live_mode = bool(hello.get("live_mode", True))

    source: LiveQueueSource = LiveQueueSource()

    async def _send(payload: dict) -> None:
        await ws.send_json(payload)

    engine_task = asyncio.create_task(run_live_turn(session_id, source, emit_event=_send, live_mode=live_mode))

    async def _reader() -> None:
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=_WS_IDLE_TIMEOUT_S)
                except (asyncio.TimeoutError, TimeoutError):
                    # Client went quiet:
                    if not source._last_text.strip():
                        # No text received during idle timeout; cancel rather than running a phantom turn
                        engine_task.cancel()
                        return
                    await source.finish()
                    return
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "chunk":
                    await source.push_text(str(msg.get("text", ""))[:2000])
                elif mtype == "end":
                    await source.finish()
                    return
                elif mtype == "cancel":
                    engine_task.cancel()
                    await source.finish()
                    try:
                        await ws.send_json({"type": "error", "message": "Turn cancelled by user."})
                    except Exception:
                        pass
                    return
        except WebSocketDisconnect:
            pass
        finally:
            # The engine must always observe utterance_end.
            await source.finish()

    reader_task = asyncio.create_task(_reader())
    try:
        await engine_task
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning("ws engine task failed: %s", e)
        try:
            await ws.send_json({"type": "error", "message": "turn failed"})
        except Exception:
            pass
    finally:
        if not reader_task.done():
            reader_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
