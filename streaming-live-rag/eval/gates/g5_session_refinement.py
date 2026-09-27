"""
Gate 5 — Session Refinement (G5).

Verifies the ORGANIZER's G5 criterion — "late-arriving constraints narrow or
update existing responses without clearing session state or re-executing
full-corpus search" — with REAL two-turn runs through run_live_turn():

  late_detail (ld_01):
    * turn 1 -> turn 2 in ONE session: answer_version goes 1 -> 2
    * turn-1 citations are a subset of turn-2 citations (preserved, not reset)
    * an injected retrieve_fn wrapper records every retrieval query: turn-2
      queries must NOT re-search the base query (no full-corpus re-search)
      and must target the delta constraint (delta_keywords)

  suppression (sup_01):
    * turn 2 classifies PRESENTATION_ONLY: retrieval_required == false,
      no retrieval_events, no new citations (subset of turn-1), version
      unchanged

SKIP (exit 2) when GROQ_API_KEY / GEMINI_API_KEY are absent, or when Qdrant
with the indexed corpus is unreachable. Target: verified state continuity.
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
    banner, footer, require_env, gate_main, GateSkipped, retrieval_preflight,
    STATUS_PASS, STATUS_FAIL,
)
from eval.dataset_loader import load_labeled_set
from streaming.engine import run_live_turn, EngineDeps, _same_intent
from streaming.live_stream import play_utterance


def _run(session_id: str, utterance: str, wpc: int, mpc: int, deps: EngineDeps):
    return asyncio.run(run_live_turn(
        session_id,
        play_utterance(utterance, words_per_chunk=wpc, ms_per_chunk=mpc),
        deps=deps,
        sink=lambda e: None,
    ))


def _eval_late_detail(sc, deps_factory):
    exp = sc["expected"]
    calls: list = []          # (turn_no, query_text)
    turn_no = [1]

    base_deps = EngineDeps.defaults()
    real_retrieve = base_deps.retrieve_fn

    def counting_retrieve(*args, **kwargs):
        q = args[0] if args else kwargs.get("query", "")
        calls.append((turn_no[0], q))
        return real_retrieve(*args, **kwargs)

    deps = deps_factory()
    deps.retrieve_fn = counting_retrieve

    wpc, mpc = sc.get("words_per_chunk", 3), sc.get("ms_per_chunk", 50)
    sid = f"g5_{sc['id']}"
    r1 = _run(sid, sc["turns"][0], wpc, mpc, deps)
    t1 = r1.telemetry
    turn_no[0] = 2
    r2 = _run(sid, sc["turns"][1], wpc, mpc, deps)
    t2 = r2.telemetry

    checks = []
    v1, v2 = t1.answer_version, t2.answer_version
    checks.append(("answer_version 1 -> 2", v1 == 1 and v2 == 2,
                   f"got v1={v1}, v2={v2}"))

    c1, c2 = set(t1.citations), set(t2.citations)
    checks.append(("turn-1 citations preserved in turn-2", c1 <= c2,
                   f"turn1={sorted(c1)} turn2={sorted(c2)}"))

    turn2_queries = [q for (tn, q) in calls if tn == 2]
    base_q = sc["turns"][0]
    re_search = [q for q in turn2_queries if _same_intent(q, base_q)]
    checks.append(("no full-corpus re-search of base query in turn 2",
                   not re_search and len(turn2_queries) > 0,
                   f"turn2 queries: {turn2_queries}"))

    kws = [k.lower() for k in exp.get("delta_keywords", [])]
    delta_hit = [q for q in turn2_queries
                 if any(k in q.lower() for k in kws)]
    checks.append(("turn-2 retrieval targets the delta constraint",
                   bool(delta_hit),
                   f"delta keywords {kws} hit in: {delta_hit}"))

    checks.append(("same session, no reset (turn_id increments)",
                   t2.turn_id == t1.turn_id + 1,
                   f"turn_ids {t1.turn_id} -> {t2.turn_id}"))
    return checks


def _eval_suppression(sc, deps_factory):
    exp = sc["expected"]
    deps = deps_factory()
    wpc, mpc = sc.get("words_per_chunk", 3), sc.get("ms_per_chunk", 50)
    sid = f"g5_{sc['id']}"
    r1 = _run(sid, sc["setup_utterance"], wpc, mpc, deps)
    t1 = r1.telemetry
    r2 = _run(sid, sc["utterance"], wpc, mpc, deps)
    t2 = r2.telemetry

    checks = []
    checks.append(("retrieval_required == false",
                   t2.retrieval_required is False,
                   f"got retrieval_required={t2.retrieval_required}"))
    checks.append(("no retrieval_events on turn 2",
                   len(t2.retrieval_events) == 0,
                   f"got {len(t2.retrieval_events)} events"))
    c1, c2 = set(t1.citations), set(t2.citations)
    checks.append(("no new citations fabricated (turn2 ⊆ turn1)",
                   c2 <= c1, f"turn1={sorted(c1)} turn2={sorted(c2)}"))
    checks.append(("answer_version unchanged by presentation turn",
                   t2.answer_version == t1.answer_version,
                   f"v {t1.answer_version} -> {t2.answer_version}"))
    checks.append(("non-empty reformatted answer", bool(r2.answer.strip()),
                   f"answer length {len(r2.answer)}"))
    return checks


def evaluate_g5():
    require_env("GROQ_API_KEY")
    require_env("GEMINI_API_KEY")
    retrieval_preflight()
    data = load_labeled_set()
    late = data.get("late_detail", [])
    suppr = data.get("suppression", [])
    if not late and not suppr:
        raise GateSkipped("no late_detail/suppression scenarios in labeled_set.yaml")

    banner("GATE 5 — Session Refinement: refine, don't restart (G5)")

    all_checks = []
    for sc in late:
        print(f"\n  late_detail {sc['id']}:")
        for name, ok, detail in _eval_late_detail(sc, EngineDeps.defaults):
            all_checks.append(ok)
            print(f"    [{'OK ' if ok else 'BAD'}] {name}\n          {detail}")
    for sc in suppr:
        print(f"\n  suppression {sc['id']}:")
        for name, ok, detail in _eval_suppression(sc, EngineDeps.defaults):
            all_checks.append(ok)
            print(f"    [{'OK ' if ok else 'BAD'}] {name}\n          {detail}")

    passed = sum(all_checks)
    total = len(all_checks)
    ok = total > 0 and passed == total
    footer(ok, "GATE 5 — verified state continuity")
    metric = f"{passed}/{total} checks"
    return (STATUS_PASS if ok else STATUS_FAIL), metric


if __name__ == "__main__":
    sys.exit(gate_main("G5", evaluate_g5))
