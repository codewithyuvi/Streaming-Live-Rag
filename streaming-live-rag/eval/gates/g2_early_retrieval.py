"""
Gate 2 — Early Retrieval (G2).

Measures REAL behavior: each streaming_early_retrieval scenario from
eval/labeled_set.yaml is replayed through run_live_turn() + play_utterance()
(real wall-clock pacing), using the REAL controller decide_fn.

Pass per scenario:
  * provisional_fired_s is not None, and
  * provisional_fired_s < utterance_end_s   (both measured, never arithmetic)
  * retrieval_events trigger taxonomy covers the expected triggers

False-trigger check: chit_chat + suppression utterances must produce NO
provisional retrieval.

SKIP (exit 2) when GROQ_API_KEY is absent — never a fake pass/fail.
Threshold: >= 80% of eligible scenarios early.
"""

import asyncio
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

from eval.gates._common import (
    banner, footer, require_env, gate_main, GateSkipped,
    STATUS_PASS, STATUS_FAIL,
)
from eval.dataset_loader import load_labeled_set
from streaming.engine import run_live_turn, EngineDeps
from streaming.live_stream import play_utterance
from controller.decide import decide_retrieval
from controller.decompose import decompose_query

THRESHOLD = 80.0


def _eval_deps() -> EngineDeps:
    """Real controller and decomposer; everything else stubbed so G2 measures the
    controller + timing + intent trigger taxonomy (no Qdrant, no synthesis LLM needed)."""
    return EngineDeps(
        decide_fn=decide_retrieval,
        decompose_fn=decompose_query,
        retrieve_fn=lambda *a, **k: [],
        classify_fn=lambda **k: {"type": "NEW_TOPIC", "reason": "g2_eval"},
        synthesize_fn=lambda p: {"text": "", "input_tokens": 0, "output_tokens": 0},
    )


def _run_turn(session_id: str, utterance: str, wpc: int, mpc: int, deps: EngineDeps):
    events: list = []

    async def _collect(ev: dict):
        events.append(ev)

    res = asyncio.run(run_live_turn(
        session_id,
        play_utterance(utterance, words_per_chunk=wpc, ms_per_chunk=mpc),
        deps=deps,
        emit_event=_collect,
        sink=lambda e: None,  # don't pollute the real telemetry log
    ))
    return res, events


def _wait_detail(scenario, controller_events) -> tuple[bool, str]:
    """Did a 'wait' decision on the early incomplete chunk precede the first
    'retrieve_now'? Reported detail (the organizer's G2 target is early
    commencement + low false triggers, not this exact sequence)."""
    exp = scenario["expected"]
    first_retrieve = next(
        (e for e in controller_events
         if e.get("type") == "controller" and e.get("trigger") == "retrieve_now"),
        None)
    if first_retrieve is None:
        return False, "no retrieve_now decision recorded"
    wait_before = [
        e for e in controller_events
        if e.get("type") == "controller" and e.get("trigger") == "wait"
        and e.get("t_s", 0) < first_retrieve.get("t_s", 0)
    ]
    chunk_ok = any(exp["wait_chunk_contains"].lower() in (e.get("query") or "").lower()
                   for e in wait_before)
    return bool(wait_before), (
        f"wait on {exp['wait_chunk_contains']!r}: {'yes' if chunk_ok else 'no'} "
        f"({len(wait_before)} wait decision(s) before first retrieve_now)"
    )


def evaluate_g2():
    require_env("GROQ_API_KEY")
    data = load_labeled_set()
    scenarios = data.get("streaming_early_retrieval", [])
    controls = data.get("chit_chat", []) + data.get("suppression", [])
    if not scenarios:
        raise GateSkipped("no streaming_early_retrieval scenarios in labeled_set.yaml")

    banner("GATE 2 — Early Retrieval (measured wall-clock) (G2)")
    deps = _eval_deps()

    passed = 0
    details = []
    for sc in scenarios:
        exp = sc["expected"]
        try:
            res, events = _run_turn(f"g2_{sc['id']}", sc["utterance"],
                                    sc.get("words_per_chunk", 2),
                                    sc.get("ms_per_chunk", 300), deps)
        except Exception as e:
            details.append({"id": sc["id"], "ok": False, "note": f"turn error: {e}"})
            continue
        t = res.telemetry
        early = (t.provisional_fired_s is not None
                 and t.provisional_fired_s < t.utterance_end_s)
        triggers = sorted({e.trigger for e in t.retrieval_events})
        taxonomy_ok = all(x in triggers for x in exp.get("retrieval_triggers", []))
        ctrl_events = [e for e in events if e.get("type") == "controller"]
        wait_ok, wait_note = _wait_detail(sc, ctrl_events)
        ok = bool(early and taxonomy_ok)
        passed += ok
        details.append({
            "id": sc["id"], "ok": ok, "early": early,
            "t_fire": t.provisional_fired_s, "t_end": t.utterance_end_s,
            "triggers": triggers, "taxonomy_ok": taxonomy_ok,
            "wait_ok": wait_ok, "wait_note": wait_note,
        })

    # False-trigger controls: no provisional retrieval may fire.
    false_triggers = 0
    control_details = []
    for cc in controls:
        utt = cc.get("utterance") or cc.get("setup_utterance")
        try:
            res, _ = _run_turn(f"g2_ctrl_{cc['id']}", utt,
                               cc.get("words_per_chunk", 2),
                               cc.get("ms_per_chunk", 300), deps)
        except Exception as e:
            control_details.append({"id": cc["id"], "false": True, "note": f"turn error: {e}"})
            false_triggers += 1
            continue
        t = res.telemetry
        fired = t.provisional_fired_s is not None
        false_triggers += fired
        control_details.append({"id": cc["id"], "false": fired,
                                "t_fire": t.provisional_fired_s})

    total = len(scenarios)
    early_rate = 100.0 * passed / total if total else 0.0
    ft_rate = 100.0 * false_triggers / len(controls) if controls else 0.0

    print(f"\nEarly retrieval: {early_rate:.1f}% ({passed}/{total}) — target >= {THRESHOLD:.0f}%")
    print(f"False-trigger rate: {ft_rate:.1f}% ({false_triggers}/{len(controls)}) — target low")
    print("-" * 70)
    for d in details:
        icon = "EARLY" if d.get("ok") else "LATE "
        print(f"  [{icon}] {d['id']}: t_fire={d.get('t_fire')} < t_end={d.get('t_end')} | "
              f"triggers={d.get('triggers')} taxonomy_ok={d.get('taxonomy_ok')}")
        print(f"          {d.get('wait_note') or d.get('note')}")
    print("-" * 70)
    for c in control_details:
        tag = "FALSE-TRIGGER" if c["false"] else "suppressed "
        extra = f" t_fire={c['t_fire']}" if c["false"] else ""
        print(f"  [{tag}] {c['id']}{extra}")

    ok = early_rate >= THRESHOLD and false_triggers == 0
    footer(ok, "GATE 2")
    metric = f"{early_rate:.1f}% early, {ft_rate:.1f}% false-trigger"
    return (STATUS_PASS if ok else STATUS_FAIL), metric


if __name__ == "__main__":
    sys.exit(gate_main("G2", evaluate_g2))
