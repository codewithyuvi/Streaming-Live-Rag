"""
Gate 6 — Telemetry & Observability (G6).

Runs ONE real turn through run_live_turn() with injected fake dependencies
(hermetic — no Qdrant, no network, no API keys) and asserts:

  1. The returned TelemetryEvent contains 100% of the required fields
     (session_id, turn_id, controller_decisions, retrieval_events,
     sub_queries, answer, citations, answer_version, latencies_ms.*,
     token_cost, utterance_end_s) with utterance-relative timestamps.
  2. The REAL telemetry sink actually wrote a fresh telemetry.jsonl line
     during THIS run (matched by unique session_id + turn_id) — never a
     stale committed file.

Target: 100% trace coverage. Always runnable (no skips).
"""

import asyncio
import json
import os
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval.gates._common import banner, footer, gate_main, STATUS_PASS, STATUS_FAIL
from streaming.engine import run_live_turn, EngineDeps
from streaming.live_stream import play_utterance
from telemetry.sink import TELEMETRY_LOG_FILE, emit as real_emit

REQUIRED_FIELDS = [
    "session_id", "turn_id", "controller_decisions", "retrieval_events",
    "sub_queries", "answer", "citations", "answer_version",
    "latencies_ms", "token_cost", "utterance_end_s",
]
REQUIRED_LATENCY_KEYS = [
    "retrieval", "rerank", "decompose", "refinement", "grounding",
    "time_to_first_token", "end_to_end", "controller", "synthesis",
    "retrieval_pipeline",
]


def _fake_deps() -> EngineDeps:
    def _hits(tag: str):
        return [(0.9, {"id": tag, "payload": {
            "text": f"Evidence for {tag}.", "tag": tag,
            "doc_id": tag.split()[0], "section": tag.split("§")[1]}}) ]

    return EngineDeps(
        decide_fn=lambda c: {"trigger": "retrieve_now", "reason": "g6_eval"},
        decompose_fn=lambda u: [
            {"sub_query": "venue capacity for 30 attendees in Pune", "intent": "capacity"},
            {"sub_query": "cancellation terms and refund policies", "intent": "cancellation"},
        ],
        retrieve_fn=lambda *a, **k: _hits("Doc_01 §1"),
        classify_fn=lambda **k: {"type": "NEW_TOPIC", "reason": "g6_eval"},
        synthesize_fn=lambda p: {
            "text": ("The maximum capacity of the Pune workshop venue is 30 attendees "
                     "[Doc_01 §1]. Standing room is not permitted [Doc_01 §1]."),
            "input_tokens": 120, "output_tokens": 45},
    )


def _fresh_log_lines(path: str, offset: int) -> list:
    with open(path, "r", encoding="utf-8") as f:
        f.seek(offset)
        return [ln for ln in f.read().splitlines() if ln.strip()]


def evaluate_g6():
    banner("GATE 6 — Telemetry & Observability (G6)")

    session_id = f"g6_{uuid.uuid4().hex[:8]}"
    before_size = (os.path.getsize(TELEMETRY_LOG_FILE)
                   if os.path.exists(TELEMETRY_LOG_FILE) else 0)

    events: list = []

    async def _collect(ev: dict):
        events.append(ev)

    res = asyncio.run(run_live_turn(
        session_id,
        play_utterance("What is the maximum capacity of the Pune workshop venue?",
                       words_per_chunk=2, ms_per_chunk=30),
        deps=_fake_deps(),
        emit_event=_collect,
        sink=real_emit,  # the REAL sink must persist this run's event
    ))
    t = res.telemetry
    dumped = t.model_dump()

    checks = []
    for f in REQUIRED_FIELDS:
        checks.append((f"field '{f}' present", f in dumped and dumped[f] is not None))
    lat = dumped.get("latencies_ms", {}) or {}
    for k in REQUIRED_LATENCY_KEYS:
        checks.append((f"latencies_ms.{k} present", k in lat))
    tc = dumped.get("token_cost", {}) or {}
    checks.append(("token_cost.usd_estimate present", "usd_estimate" in tc))

    # Utterance-relative timestamps (never epoch seconds).
    ts_ok = True
    for ev in dumped.get("retrieval_events", []) or []:
        if ev.get("timestamp_s", 0) > 1000:
            ts_ok = False
    for d in dumped.get("controller_decisions", []) or []:
        if d.get("timestamp_s", 0) > 1000:
            ts_ok = False
    if dumped.get("utterance_end_s", 0) > 1000:
        ts_ok = False
    checks.append(("timestamps utterance-relative (<1000s)", ts_ok))

    # Event stream ordering sanity.
    order = [e.get("type") for e in events]
    expected_seq = ["stream_started", "transcript", "controller",
                    "retrieval_started", "utterance_end", "answer", "telemetry"]
    pos, seq_ok = 0, True
    for want in expected_seq:
        try:
            pos = order.index(want, pos) + 1
        except ValueError:
            seq_ok = False
            break
    checks.append(("live event order stream_started→…→telemetry", seq_ok))

    # Freshness: the real sink must have written THIS run's event.
    fresh_ok, fresh_detail = False, "no new line matched this run"
    if os.path.exists(TELEMETRY_LOG_FILE):
        for ln in _fresh_log_lines(TELEMETRY_LOG_FILE, before_size):
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if rec.get("session_id") == session_id and rec.get("turn_id") == t.turn_id:
                fresh_ok = True
                fresh_detail = f"matched session_id={session_id} turn_id={t.turn_id}"
                break
    checks.append((f"telemetry.jsonl written fresh during run ({fresh_detail})", fresh_ok))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"\nTrace coverage: {100.0 * passed / total:.1f}% ({passed}/{total}) — target 100%")
    print("-" * 70)
    for name, ok in checks:
        print(f"  [{'OK ' if ok else 'MISS'}] {name}")

    ok = passed == total
    footer(ok, "GATE 6")
    return (STATUS_PASS if ok else STATUS_FAIL), f"{100.0 * passed / total:.1f}% ({passed}/{total})"


if __name__ == "__main__":
    sys.exit(gate_main("G6", evaluate_g6))
