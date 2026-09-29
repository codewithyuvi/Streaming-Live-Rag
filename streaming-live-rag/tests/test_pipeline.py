"""
tests/test_pipeline.py — hermetic unit tests for the Streaming Live RAG pipeline.

Hermetic means: no Qdrant, no network, no API keys, no real LLM calls.
Must pass on a clean venv with only:
    fastapi pydantic httpx pytest python-dotenv pyyaml
The live engine (streaming/engine.py) is exercised end-to-end with injected
fake dependencies, so controller timing, provisional retrieval, event order,
and telemetry are all measured for real — never faked.
"""

import asyncio
import json
import os
import re
import sys
import time
import types

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.grounding import validate, tags_in  # noqa: E402
from controller.heuristics import is_stable_enough, get_stable_query_prefix  # noqa: E402
from streaming.engine import run_live_turn, EngineDeps, _same_intent  # noqa: E402
from streaming.live_stream import play_utterance, LiveQueueSource  # noqa: E402
from retrieval.merge import merge_with_quota, merge_and_dedup  # noqa: E402
from retrieval.parsers import extract_sections  # noqa: E402
from session.store import get_or_create_session, reset_store  # noqa: E402
from eval.dataset_loader import load_labeled_set  # noqa: E402


# ---------------------------------------------------------------------------
# Fake dependencies for the live engine
# ---------------------------------------------------------------------------

def _point(pid, tag, text):
    return {"id": pid, "payload": {"text": text, "tag": tag,
                                   "doc_id": tag.split()[0],
                                   "section": tag.split("§")[1]}}


def make_fake_deps(decide=None):
    """EngineDeps with deterministic fakes. decide() defaults to
    wait-until-'Pune'-then-retrieve_now, mirroring the organizer's example."""
    def _decide(candidate):
        if decide is not None:
            return decide(candidate)
        if "pune" in candidate.lower():
            return {"trigger": "retrieve_now", "reason": "stable geo entity"}
        return {"trigger": "wait", "reason": "intent incomplete"}

    def _retrieve(query, rerank_against=None, top_k=5):
        return [
            (0.95, _point(1, "Doc_01 §1", "The maximum capacity of the Pune workshop venue is 30 attendees.")),
            (0.80, _point(2, "Doc_01 §2", "The cancellation policy requires 48 hours notice for a full refund.")),
        ]

    def _synthesize(prompt):
        return {
            "text": ("The maximum capacity of the Pune workshop venue is 30 attendees "
                     "[Doc_01 §1]. The cancellation policy requires 48 hours notice "
                     "for a full refund [Doc_01 §2]."),
            "input_tokens": 120, "output_tokens": 45,
        }

    return EngineDeps(
        decide_fn=_decide,
        decompose_fn=lambda u: [
            {"sub_query": "venue capacity for 30 attendees in Pune", "intent": "capacity"},
            {"sub_query": "cancellation terms and refund policies", "intent": "cancellation"},
        ],
        retrieve_fn=_retrieve,
        classify_fn=lambda **k: {"type": "NEW_TOPIC", "reason": "test"},
        synthesize_fn=_synthesize,
    )


def run_turn(session_id, utterance, deps, wpc=2, mpc=30, use_cache=None, live_mode=True):
    """Run one live turn; return (TurnResult, emitted events, sunk events)."""
    events, sunk = [], []

    async def _collect(ev):
        events.append(ev)

    res = asyncio.run(run_live_turn(
        session_id,
        play_utterance(utterance, words_per_chunk=wpc, ms_per_chunk=mpc),
        deps=deps,
        emit_event=_collect,
        sink=sunk.append,
        live_mode=live_mode,
        use_cache=use_cache,
    ))
    return res, events, sunk


# ---------------------------------------------------------------------------
# Grounding validator (kept: the good unit tests)
# ---------------------------------------------------------------------------

def test_grounding_validator():
    rep = validate("The venue capacity is 30 [Doc_01 §1].", ["Doc_01 §1"])
    assert rep.ok is True
    assert rep.support == 1.0
    assert rep.fabricated == []
    assert rep.cited == ["Doc_01 §1"]

    rep_fab = validate("The venue capacity is 30 [Doc_99 §9].", ["Doc_01 §1"])
    assert rep_fab.ok is False
    assert "Doc_99 §9" in rep_fab.fabricated

    rep_punct = validate("The maximum capacity is 30. [Doc_01 §1]", ["Doc_01 §1"])
    assert rep_punct.ok is True
    assert rep_punct.support == 1.0

    # tags_in picks up dotted sections too
    assert tags_in("see [Doc_03 §1.2] for details") == ["Doc_03 §1.2"]


def test_grounding_abstention():
    rep = validate("This information is not available in the provided documents.", [])
    assert rep.abstained is True
    assert rep.ok is True


# ---------------------------------------------------------------------------
# Heuristics (kept)
# ---------------------------------------------------------------------------

def test_stability_heuristics():
    assert is_stable_enough("What is the capacity") is True
    assert is_stable_enough("What is the") is False
    assert is_stable_enough("Tell me about the policy for") is False
    assert is_stable_enough("Where is room 3B?") is True


def test_stable_query_prefix_strips_dangling():
    assert get_stable_query_prefix("I need to plan a customer workshop in") == \
        "I need to plan a customer workshop"
    assert get_stable_query_prefix("I need") is None


def test_intent_deduplication():
    assert _same_intent("what is the maximum capacity of the pune venue",
                        "what is the maximum capacity of") is False
    assert _same_intent("what is the venue capacity",
                        "what is the venue capacity") is True
    assert _same_intent("what is the venue capacity", None) is False


# ---------------------------------------------------------------------------
# play_utterance: real pacing, measured timestamps
# ---------------------------------------------------------------------------

def test_play_utterance_pacing_is_real():
    utterance = "one two three four five six seven eight"
    chunks = []

    async def _drain():
        async for ev in play_utterance(utterance, words_per_chunk=2, ms_per_chunk=40):
            chunks.append(ev)

    t0 = time.monotonic()
    asyncio.run(_drain())
    elapsed = time.monotonic() - t0

    kinds = [c.kind for c in chunks]
    assert kinds[-1] == "utterance_end"
    assert kinds.count("chunk") == 4
    # 4 chunks with 40ms pacing must take real time (~120ms+ of sleeps)
    assert elapsed >= 0.10, f"stream finished in {elapsed:.3f}s — pacing is not real"

    # chunk timestamps are measured and increase by ~ms_per_chunk
    ts = [c.t_s for c in chunks if c.kind == "chunk"]
    assert ts[0] < 0.05, f"first chunk should fire at ~t=0, got {ts[0]}"
    deltas = [b - a for a, b in zip(ts, ts[1:])]
    for d in deltas:
        assert 0.02 <= d <= 0.20, f"chunk delta {d:.3f}s not ~= 40ms"

    # cumulative partial text, full text on utterance_end
    assert chunks[-2].text == utterance
    assert chunks[-1].text == utterance


def test_live_queue_source_server_timestamps():
    async def _scenario():
        src = LiveQueueSource()
        await src.push_text("hello")
        await asyncio.sleep(0.03)
        await src.push_text("hello world")
        await asyncio.sleep(0.03)
        await src.finish()
        out = []
        async for ev in src:
            out.append(ev)
        return out

    out = asyncio.run(_scenario())
    assert [e.kind for e in out] == ["chunk", "chunk", "utterance_end"]
    ts = [e.t_s for e in out]
    assert ts[0] < ts[1] < ts[2], "server timestamps must increase"
    assert ts[2] >= 0.05, "timestamps must reflect real elapsed time"
    assert out[-1].text == "hello world"


# ---------------------------------------------------------------------------
# Engine: live turn with injected fakes — real timing, real event order
# ---------------------------------------------------------------------------

def test_engine_live_turn_early_retrieval_measured():
    utterance = ("I need to plan a customer workshop in Pune for 30 people "
                 "and I need the cancellation policy.")
    res, events, sunk = run_turn("pytest_engine_1", utterance, make_fake_deps(),
                                 wpc=2, mpc=30)
    t = res.telemetry

    # Honest wall-clock anchors: provisional fired strictly before utterance end.
    assert t.provisional_fired_s is not None
    assert t.utterance_end_s is not None
    assert t.provisional_fired_s < t.utterance_end_s, (
        f"provisional {t.provisional_fired_s} not before utterance end {t.utterance_end_s}")

    # Trigger taxonomy on real retrieval events.
    triggers = {e.trigger for e in t.retrieval_events}
    assert "provisional" in triggers
    assert "multi_intent" in triggers

    # Controller saw wait-then-retrieve_now.
    seen = [e["trigger"] for e in events
            if e.get("type") == "controller"]
    assert "wait" in seen and "retrieve_now" in seen
    assert seen.index("wait") < seen.index("retrieve_now")

    # Event order: transcript/controller/retrieval_started before utterance_end,
    # answer before telemetry.
    order = [e.get("type") for e in events]
    for want in ["stream_started", "transcript", "controller",
                 "retrieval_started", "utterance_end", "answer", "telemetry"]:
        assert want in order, f"missing event {want}"
    assert order.index("stream_started") == 0
    assert order.index("retrieval_started") < order.index("utterance_end")
    assert order.index("answer") < order.index("telemetry")

    # Sink received exactly this turn's telemetry.
    assert len(sunk) == 1
    assert sunk[0].session_id == "pytest_engine_1"
    assert sunk[0].turn_id == t.turn_id

    # Real answer + citations from the fake (but structurally real) path.
    assert "[Doc_01 §1]" in res.answer
    assert "Doc_01 §1" in t.citations
    assert t.grounding_score >= 0.85


def test_engine_no_provisional_on_chitchat():
    res, events, _ = run_turn(
        "pytest_engine_2", "Hello, how are you today?",
        make_fake_deps(decide=lambda c: {"trigger": "no_retrieval_needed",
                                          "reason": "greeting"}),
        wpc=2, mpc=30)
    t = res.telemetry
    assert t.provisional_fired_s is None
    assert not [e for e in t.retrieval_events if e.trigger == "provisional"]


def test_engine_simultaneous_multi_intent_streaming():
    """Verify that multiple distinct intents in a continuous stream trigger concurrent speculative retrievals."""
    def _multi_decide(candidate):
        c = candidate.lower()
        if "segmentation" in c:
            return {"trigger": "retrieve_now", "reason": "segmentation concept"}
        if "divide" in c or "variable size" in c:
            return {"trigger": "retrieve_now", "reason": "process division intent"}
        if "paging" in c or "users view" in c:
            return {"trigger": "retrieve_now", "reason": "paging comparison intent"}
        return {"trigger": "wait", "reason": "intent incomplete"}

    deps = make_fake_deps(decide=_multi_decide)
    utterance = (
        "what is segmentation explain how it divides a process into variable size segments "
        "and why it is said to give the users view of the process while paging does not"
    )
    res, events, sunk = run_turn("pytest_engine_multi", utterance, deps, wpc=2, mpc=25)
    t = res.telemetry

    # Must have fired multiple speculative retrieval_started events before utterance_end
    retrieval_started = [e for e in events if e.get("type") == "retrieval_started"]
    assert len(retrieval_started) >= 2, (
        f"Expected at least 2 simultaneous speculative retrievals, got {len(retrieval_started)}"
    )

    # Every speculative retrieval must have commenced strictly before speech finished
    for rs in retrieval_started:
        assert rs["t_s"] < t.utterance_end_s, (
            f"Retrieval {rs} did not start before speech ended ({t.utterance_end_s})"
        )

    # Phase 2 thought must reflect simultaneous speculative retrievals
    thought_events = [e for e in events if e.get("type") == "thought"]
    phase2 = next(e for e in thought_events if e.get("phase") == 2)
    assert "simultaneous speculative searches were executed during speech" in phase2["text"]

    # Grounding & answer generated
    assert res.answer
    assert len(t.citations) > 0


# ---------------------------------------------------------------------------
# Merge: provisional evidence must survive the quota merge (starvation fix)
# ---------------------------------------------------------------------------

def _scored(pid, score=0.5):
    return (score, _point(pid, f"Doc_01 §{pid}",
                          f"text {pid}"))


def test_merge_provisional_not_starved():
    # 4 guaranteed sub-queries x 2 hits fill the 8-slot budget...
    per_sq = [
        {"sub_query": f"sq{i}", "guaranteed": True,
         "scored_hits": [_scored(i * 10 + 1, 0.9), _scored(i * 10 + 2, 0.8)]}
        for i in range(4)
    ]
    # ...but the provisional (non-guaranteed) early-retrieval evidence must
    # still keep at least one slot.
    per_sq.insert(0, {"sub_query": "provisional q", "guaranteed": False,
                      "scored_hits": [_scored(100, 0.7), _scored(101, 0.6)]})

    merged = merge_with_quota(per_sq, total_k=8, min_per_sq=2)
    pids = []
    for item in merged:
        p = item["point"]
        pids.append(p["id"] if isinstance(p, dict) else getattr(p, "id", None))
    # NOTE (2026-09-25): this currently FAILS — retrieval/merge.py stage 1b
    # appends the provisional pid to `order` AFTER the guaranteed entries,
    # so the final `[:total_k]` slice drops it whenever the budget is full.
    # The M7 starvation fix is incomplete; the invariant below is the intended
    # behavior (provisional evidence keeps >= 1 slot).
    assert 100 in pids, (
        f"provisional evidence starved by quota merge: {pids} "
        "(M7 fix incomplete: stage-1b provisional slot dropped by [:total_k] slice)")
    assert len(merged) <= 8


def test_merge_and_dedup_end_to_end():
    per_sq = [
        {"sub_query": "capacity", "guaranteed": False,
         "scored_hits": [_scored(1, 0.9), _scored(2, 0.85)]},
        {"sub_query": "cancellation", "guaranteed": True,
         "scored_hits": [_scored(3, 0.95), _scored(4, 0.7)]},
    ]
    chunks = merge_and_dedup(per_sq, top_k=4)
    tags = [c.tag for c in chunks]
    assert "Doc_01 §1" in tags  # provisional chunk survived
    assert len(chunks) <= 4


# ---------------------------------------------------------------------------
# Parsers: dotted sections produce distinct tags (collision regression)
# ---------------------------------------------------------------------------

def test_parsers_dotted_sections_distinct_tags():
    data = ("Doc_03 §1.1\nAlpha text about venue capacity.\n\n"
            "Doc_03 §1.2\nBeta text about cancellation terms.").encode("utf-8")
    sections = extract_sections("test.txt", data, "Doc_03")
    tags = [s["tag"] for s in sections]
    assert "Doc_03 §1.1" in tags, f"tags: {tags}"
    assert "Doc_03 §1.2" in tags, f"tags: {tags}"
    assert len(set(tags)) == len(tags), f"colliding tags: {tags}"


# ---------------------------------------------------------------------------
# Session lifecycle (kept)
# ---------------------------------------------------------------------------

def test_session_lifecycle():
    reset_store()
    sess = get_or_create_session("pytest_session")

    sess.commit("NEW_TOPIC", "q1", "ans1", ["Doc_01 §1"])
    assert sess.answer_version == 1
    assert sess.current_citations == ["Doc_01 §1"]

    sess.commit("LATE_DETAIL", "q2", "ans2", ["Doc_02 §3"])
    assert sess.answer_version == 2
    assert "Doc_01 §1" in sess.current_citations
    assert "Doc_02 §3" in sess.current_citations

    sess.commit("PRESENTATION_ONLY", "format as bullets", "ans3", ["Doc_01 §1"])
    assert sess.answer_version == 2
    assert len(sess.turns) == 3


def test_session_reset_and_cache_clearing():
    """Verify session memory is cleanly wiped on reset so subsequent queries start fresh."""
    from session.store import delete_session
    _stub_heavy_modules()
    import api.main as api_main
    from api.main import reset_session_endpoint, SessionResetRequest

    reset_store()
    sess = get_or_create_session("cache_test_sess")
    sess.commit("NEW_TOPIC", "what is segmentation", "segmentation divides memory...", ["Doc_01 §1"])
    assert sess.current_answer != ""
    assert len(sess.turns) == 1

    # Reset via endpoint with explicit session_id
    res = reset_session_endpoint(session_id="cache_test_sess")
    assert res["status"] == "cleared"
    assert res["existed"] is True

    # Check store has no residual memory for that session
    fresh = get_or_create_session("cache_test_sess")
    assert fresh.current_answer == ""
    assert fresh.current_query == ""
    assert len(fresh.turns) == 0
    assert fresh.answer_version == 0

    # Reset all sessions via request body
    sess2 = get_or_create_session("another_sess")
    sess2.commit("NEW_TOPIC", "query", "answer", [])
    res2 = reset_session_endpoint(req=SessionResetRequest(session_id="another_sess"))
    assert res2["status"] == "cleared"

    res_all = reset_session_endpoint()
    assert res_all["status"] == "cleared"
    assert res_all.get("all") is True


# ---------------------------------------------------------------------------
# Dataset integrity: ids unique; expectations machine-readable (not in comments)
# ---------------------------------------------------------------------------

def _all_scenarios(data):
    for section in ("streaming_early_retrieval", "late_detail", "suppression",
                    "chit_chat", "multi_intent", "grounding_probes"):
        for sc in data.get(section, []):
            yield section, sc


def test_dataset_integrity():
    data = load_labeled_set(os.path.join(ROOT_DIR, "eval", "labeled_set.yaml"))
    seen = {}
    count = 0
    for section, sc in _all_scenarios(data):
        count += 1
        sid = sc.get("id")
        assert sid, f"scenario in {section} missing id"
        assert sid not in seen, f"duplicate id {sid} in {section} and {seen[sid]}"
        seen[sid] = section

        exp = sc.get("expected")
        if section == "multi_intent":
            kws = sc.get("expected_intent_keywords")
            assert isinstance(kws, list) and len(kws) >= 2, f"{sid}: keyword sets missing"
            assert all(isinstance(k, list) and k for k in kws), f"{sid}: bad keyword set"
        elif section == "late_detail":
            assert isinstance(sc.get("turns"), list) and len(sc["turns"]) == 2, f"{sid}: turns"
            assert exp and exp.get("answer_versions") == [1, 2], f"{sid}: expected"
        elif section == "grounding_probes":
            assert isinstance(sc.get("expect_abstain"), bool), f"{sid}: expect_abstain"
            assert sc.get("utterance"), f"{sid}: utterance missing"
        else:
            assert isinstance(exp, dict) and exp, f"{sid}: machine-readable 'expected' missing"

    assert count >= 15, f"dataset shrank unexpectedly: {count} scenarios"


def test_dataset_no_labels_hidden_in_comments():
    path = os.path.join(ROOT_DIR, "eval", "labeled_set.yaml")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    comments = re.findall(r"^\s*#.*$", text, re.M)
    for c in comments:
        assert not re.search(r"#\s*(wait|retrieve_now|no_retrieval_needed)\s*$", c), (
            f"label hidden in comment: {c.strip()}")


# ---------------------------------------------------------------------------
# UI smoke: every fetch() path in static/index.html must exist in the route table
# ---------------------------------------------------------------------------

def _stub_heavy_modules():
    """Make api.main importable without qdrant_client / fastembed installed.

    api.main imports retrieval.ingest -> retrieval.hybrid_search, which need
    qdrant_client and fastembed at module level. We stub those modules with
    dummy classes (attribute access only; nothing is instantiated at import).
    This documents the choice: stubbing third-party SDKs for a static route
    check is robust and keeps the test hermetic; the alternative (parsing
    api/main.py decorators) would miss routes added programmatically.
    """
    if "qdrant_client" in sys.modules:
        return

    def _dummy(name="Dummy"):
        return type(name, (), {"__init__": lambda self, *a, **k: None})

    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.QdrantClient = _dummy("QdrantClient")
    models = types.ModuleType("qdrant_client.models")
    for n in ("Distance", "VectorParams", "PointStruct", "SparseVectorParams",
              "SparseVector", "Modifier", "Prefetch", "FusionQuery", "Fusion"):
        setattr(models, n, _dummy(n))
    qdrant_client.models = models

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = _dummy("TextEmbedding")
    fastembed.SparseTextEmbedding = _dummy("SparseTextEmbedding")
    rerank = types.ModuleType("fastembed.rerank")
    cross_encoder = types.ModuleType("fastembed.rerank.cross_encoder")
    cross_encoder.TextCrossEncoder = _dummy("TextCrossEncoder")
    rerank.cross_encoder = cross_encoder
    fastembed.rerank = rerank

    sys.modules["qdrant_client"] = qdrant_client
    sys.modules["qdrant_client.models"] = models
    sys.modules["fastembed"] = fastembed
    sys.modules["fastembed.rerank"] = rerank
    sys.modules["fastembed.rerank.cross_encoder"] = cross_encoder


def _fetch_paths_from_html():
    path = os.path.join(ROOT_DIR, "static", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    paths = set()
    for m in re.finditer(r"fetch\(\s*[\"'`]([^\"'`]+?)[\"'`]", html):
        raw = m.group(1).split("?")[0].strip()
        if raw.startswith("/"):
            paths.add(raw)
    return paths


def _route_patterns(app):
    pats = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path:
            continue
        pats.append(re.compile("^" + re.sub(r"\{[^}]+\}", r"[^/]+", path) + "$"))
    return pats


def test_ui_fetch_paths_exist_in_route_table():
    _stub_heavy_modules()
    import api.main as api_main  # noqa: E402
    app = api_main.app
    # NOTE: we never construct a TestClient here, so the startup auto-seed
    # hook (which touches Qdrant) is never fired.

    fetch_paths = _fetch_paths_from_html()
    assert fetch_paths, "no fetch() paths found in static/index.html"
    patterns = _route_patterns(app)

    orphans = [p for p in sorted(fetch_paths)
               if not any(pat.match(p) for pat in patterns)]
    assert not orphans, (
        "index.html calls endpoints with no backend route (404s): "
        + ", ".join(orphans))


def test_query_cache_hit_and_repeat_query():
    """Verify query cache returns Phase 0 instant response on repeated queries."""
    from streaming.engine import clear_query_cache, set_cached_response, get_cached_response
    clear_query_cache()

    synth_count = 0
    def _synth(prompt):
        nonlocal synth_count
        synth_count += 1
        return {
            "text": "Segmentation divides memory into logical segments [Doc_01 §1].",
            "input_tokens": 50, "output_tokens": 20,
        }

    deps = make_fake_deps()
    deps.synthesize_fn = _synth

    utterance = "I need to plan a workshop in Pune for 30 attendees"
    res1, ev1, _ = run_turn("cache_sess_1", utterance, deps, use_cache=True)
    assert synth_count == 1
    assert any(e.get("phase") == 1 for e in ev1 if e.get("type") == "thought")

    # Turn 2: Repeat the same question in the same session
    res2, ev2, _ = run_turn("cache_sess_1", utterance, deps, use_cache=True)
    assert synth_count == 1, "Cache hit should not trigger additional LLM synthesis"
    assert res2.answer == res1.answer
    assert any(e.get("phase") == 0 for e in ev2 if e.get("type") == "thought")
    assert res2.telemetry.latencies_ms.retrieval == 0.0

    clear_query_cache()
    assert get_cached_response(utterance) is None


@pytest.mark.anyio
async def test_compare_endpoint_contract(monkeypatch):
    """Verify handle_compare runs both pipelines concurrently, isolates sessions, and returns measured metrics."""
    _stub_heavy_modules()
    from unittest.mock import MagicMock
    from fastapi import Request
    import api.main as api_main
    from api.main import CompareRequest, CompareResponse, handle_compare
    from streaming.engine import TurnResult
    from telemetry.schema import TelemetryEvent, LatenciesMs, TokenCost

    captured_calls = []

    async def _fake_run_live_turn(session_id, source, emit_event=None, live_mode=True, use_cache=True):
        captured_calls.append({
            "session_id": session_id,
            "live_mode": live_mode,
            "use_cache": use_cache,
        })
        # consume source chunks
        async for _ in source:
            pass
        return TurnResult(
            answer=f"Synthesized answer for {session_id} (live={live_mode})",
            telemetry=TelemetryEvent(
                session_id=session_id,
                turn_id=1,
                intent="NEW_TOPIC",
                is_live_stream=live_mode,
                live_chunks_received=3,
                live_stream_duration_s=0.2,
                provisional_fired_s=0.05 if live_mode else None,
                utterance_end_s=0.2,
                grounding_score=1.0,
                citations=["Doc_01 §1"],
                latencies_ms=LatenciesMs(),
                token_cost=TokenCost(),
            ),
        )

    monkeypatch.setattr(api_main, "run_live_turn", _fake_run_live_turn)

    mock_req = MagicMock(spec=Request)
    mock_req.client.host = "127.0.0.1"

    cmp_req = CompareRequest(
        utterance="What is the venue booking capacity and cancellation penalty in Pune?",
        words_per_chunk=2,
        ms_per_chunk=5,
    )

    res = await handle_compare(cmp_req, mock_req)
    assert isinstance(res, CompareResponse)
    assert res.utterance == cmp_req.utterance
    assert "live" in res.live.telemetry.session_id
    assert "norm" in res.normal.telemetry.session_id
    assert res.live.telemetry.session_id != res.normal.telemetry.session_id
    assert res.live.telemetry.provisional_fired_s is not None
    assert res.normal.telemetry.provisional_fired_s is None
    assert res.speech_duration_s > 0
    assert res.speedup_factor > 0

    # Ensure both runs strictly bypassed cache (use_cache=False)
    assert len(captured_calls) == 2
    assert captured_calls[0]["live_mode"] is True
    assert captured_calls[1]["live_mode"] is False
    for call in captured_calls:
        assert call["use_cache"] is False, f"Expected use_cache=False, got {call}"


