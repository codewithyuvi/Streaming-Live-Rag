"""
Gate 3 — Multi-Intent Identification (G3).

Measures whether the decomposer identifies the RIGHT intents, not just the
right count. Each multi_intent scenario in eval/labeled_set.yaml carries
expected_intent_keywords: a list of keyword sets. A sub-query matches an
intent if it contains >= 1 keyword from the set (case-insensitive).

Scenario passes when >= 2 DISTINCT intent sets are matched by the returned
sub-queries (the organizer's "at least two distinct sub-intents").

Uses the REAL decompose_fn. SKIP (exit 2) when GROQ_API_KEY is absent.
Threshold: >= 70% of compound scenarios pass.
"""

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
from controller.decompose import decompose_query

THRESHOLD = 70.0
MIN_DISTINCT_INTENTS = 2


def match_intents(sub_queries: list[str], keyword_sets: list[list[str]]) -> list[int]:
    """Return the indices of intent sets matched by at least one sub-query."""
    matched = []
    for i, kws in enumerate(keyword_sets):
        needles = [k.lower() for k in kws]
        if any(any(n in sq.lower() for n in needles) for sq in sub_queries):
            matched.append(i)
    return matched


def evaluate_g3():
    require_env("GROQ_API_KEY")
    data = load_labeled_set()
    scenarios = data.get("multi_intent", [])
    if not scenarios:
        raise GateSkipped("no multi_intent scenarios in labeled_set.yaml")

    banner("GATE 3 — Multi-Intent Identification (G3)")

    passed = 0
    details = []
    for sc in scenarios:
        try:
            raw = decompose_query(sc["utterance"])
        except Exception as e:
            details.append({"id": sc["id"], "ok": False, "note": f"decompose error: {e}"})
            continue
        sub_queries = [d.get("sub_query", "") for d in raw
                       if isinstance(d, dict) and d.get("sub_query")]
        keyword_sets = sc["expected_intent_keywords"]
        matched = match_intents(sub_queries, keyword_sets)
        ok = len(matched) >= MIN_DISTINCT_INTENTS
        passed += ok
        details.append({
            "id": sc["id"], "ok": ok,
            "sub_queries": sub_queries,
            "matched": matched, "of": len(keyword_sets),
        })

    total = len(scenarios)
    rate = 100.0 * passed / total if total else 0.0
    print(f"\nMulti-intent identification: {rate:.1f}% ({passed}/{total}) — target >= {THRESHOLD:.0f}%")
    print("-" * 70)
    for d in details:
        icon = "OK  " if d.get("ok") else "MISS"
        print(f"  [{icon}] {d['id']}: {d.get('matched')}/{d.get('of')} intents matched")
        for sq in d.get("sub_queries", []):
            print(f"          - {sq}")
        if d.get("note"):
            print(f"          {d['note']}")

    ok = rate >= THRESHOLD
    footer(ok, "GATE 3")
    return (STATUS_PASS if ok else STATUS_FAIL), f"{rate:.1f}% ({passed}/{total})"


if __name__ == "__main__":
    sys.exit(gate_main("G3", evaluate_g3))
