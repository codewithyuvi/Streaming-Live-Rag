"""
Gate 3 — Multi-Intent Identification Evaluation (G3)

Measures:
  1. G3 score: % of compound cases where the decomposer correctly identifies
     the expected number of sub-intents (±1 tolerance).
  2. Over-fragmentation rate: % of single-intent cases where the decomposer
     incorrectly produces >1 sub-query.

Target: G3 ≥ 70% on compound cases, low over-fragmentation on singles.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from controller.decompose import decompose_query
from eval.dataset_loader import load_labeled_set


def evaluate_g3():
    data = load_labeled_set()
    queries = data.get("queries", [])

    # Separate compound vs single-intent cases
    compound_cases = [q for q in queries if q.get("expected_intents") is not None and q["expected_intents"] > 1]
    single_cases = [q for q in queries if q.get("expected_intents") is not None and q["expected_intents"] == 1]

    # --- Evaluate compound cases ---
    compound_correct = 0
    compound_total = len(compound_cases)
    compound_details = []

    for q in compound_cases:
        utterance = q["utterance"]
        expected = q["expected_intents"]

        sub_queries = decompose_query(utterance)
        actual = len(sub_queries)

        # ±1 tolerance
        match = abs(actual - expected) <= 1 and actual > 1
        if match:
            compound_correct += 1

        status = "✅" if match else "❌"
        compound_details.append({
            "id": q["id"],
            "utterance": utterance[:60] + "..." if len(utterance) > 60 else utterance,
            "expected": expected,
            "actual": actual,
            "status": status,
            "sub_queries": [sq["sub_query"] for sq in sub_queries]
        })

    # --- Evaluate single-intent cases (over-fragmentation) ---
    over_frag_count = 0
    single_total = len(single_cases)
    single_details = []

    for q in single_cases:
        utterance = q["utterance"]

        sub_queries = decompose_query(utterance)
        actual = len(sub_queries)

        fragmented = actual > 1
        if fragmented:
            over_frag_count += 1

        status = "❌ OVER-FRAG" if fragmented else "✅"
        single_details.append({
            "id": q["id"],
            "utterance": utterance[:60] + "..." if len(utterance) > 60 else utterance,
            "actual": actual,
            "status": status,
            "sub_queries": [sq["sub_query"] for sq in sub_queries]
        })

    # --- Report ---
    g3_score = (compound_correct / compound_total * 100) if compound_total > 0 else 0
    over_frag_rate = (over_frag_count / single_total * 100) if single_total > 0 else 0

    print("=" * 70)
    print("    GATE 3 — Multi-Intent Identification (G3)")
    print("=" * 70)

    print(f"\n📊 G3 Score: {g3_score:.1f}% ({compound_correct}/{compound_total}) — Target: ≥ 70%")
    print(f"📊 Over-fragmentation: {over_frag_rate:.1f}% ({over_frag_count}/{single_total}) — Target: LOW")

    print(f"\n{'─' * 70}")
    print("COMPOUND CASES (should produce >1 sub-query):")
    print(f"{'─' * 70}")
    for d in compound_details:
        print(f"  {d['status']} {d['id']}: expected={d['expected']}, actual={d['actual']}")
        print(f"     Query: {d['utterance']}")
        for i, sq in enumerate(d["sub_queries"]):
            print(f"       └─ [{i+1}] {sq}")
        print()

    print(f"{'─' * 70}")
    print("SINGLE-INTENT CONTROLS (should produce exactly 1 sub-query):")
    print(f"{'─' * 70}")
    for d in single_details:
        print(f"  {d['status']} {d['id']}: actual={d['actual']}")
        print(f"     Query: {d['utterance']}")
        for i, sq in enumerate(d["sub_queries"]):
            print(f"       └─ [{i+1}] {sq}")
        print()

    # --- Pass/Fail ---
    passed = g3_score >= 70 and over_frag_rate <= 30
    print(f"{'=' * 70}")
    if passed:
        print("🟢 GATE 3 PASSED")
    else:
        print("🔴 GATE 3 FAILED")
        if g3_score < 70:
            print(f"   ↳ G3 score {g3_score:.1f}% is below 70% target")
        if over_frag_rate > 30:
            print(f"   ↳ Over-fragmentation {over_frag_rate:.1f}% exceeds 30% threshold")
    print(f"{'=' * 70}")

    return passed


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    evaluate_g3()
