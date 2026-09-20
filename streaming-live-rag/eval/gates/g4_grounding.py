"""
Gate 4 — Grounding Validation (G4)

Validates that the LLM's synthesis output is properly grounded:
  1. All citation tags in the answer exist in the provided context
  2. Zero fabricated Doc IDs or Section numbers
  3. Uncertainty is expressed when the corpus doesn't cover a query

This gate tests the grounding validator itself (not the LLM) by running
it against synthetic answer strings with known citation patterns.

Target: G4 ≥ 85% citation support, zero fabricated IDs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from retrieval.grounding import validate_grounding


def evaluate_g4():
    """
    Runs the grounding validator against a suite of synthetic test answers
    with known-good and known-bad citation patterns.
    """

    available_tags = ["Doc_01 §1", "Doc_01 §2", "Doc_02 §1", "Doc_02 §2"]

    test_cases = [
        # --- Valid cases ---
        {
            "name": "Perfect single citation",
            "answer": "The venue capacity is 30 [Doc_01 §1].",
            "expect_grounded": True,
            "expect_fabricated": 0,
        },
        {
            "name": "Multiple valid citations",
            "answer": "The capacity is 30 [Doc_01 §1]. Travel requires Director approval [Doc_02 §1].",
            "expect_grounded": True,
            "expect_fabricated": 0,
        },
        {
            "name": "All four valid tags used",
            "answer": "Venue: 30 people [Doc_01 §1], cancel 48h [Doc_01 §2]. Travel: Director [Doc_02 §1], VP exception [Doc_02 §2].",
            "expect_grounded": True,
            "expect_fabricated": 0,
        },
        {
            "name": "Valid citation with uncertainty for partial answer",
            "answer": "The capacity is 30 [Doc_01 §1]. Parking information is not available in the provided documents.",
            "expect_grounded": True,
            "expect_fabricated": 0,
        },
        {
            "name": "Pure uncertainty (no citations needed)",
            "answer": "This information is not available in the provided documents.",
            "expect_grounded": True,
            "expect_fabricated": 0,
        },
        # --- Fabrication cases ---
        {
            "name": "Fabricated Doc ID",
            "answer": "The answer is X [Doc_03 §1].",
            "expect_grounded": False,
            "expect_fabricated": 1,
        },
        {
            "name": "Fabricated Section number",
            "answer": "The answer is X [Doc_01 §5].",
            "expect_grounded": False,
            "expect_fabricated": 1,
        },
        {
            "name": "Mix of valid and fabricated",
            "answer": "Capacity is 30 [Doc_01 §1]. Also see [Doc_99 §3].",
            "expect_grounded": False,
            "expect_fabricated": 1,
        },
        {
            "name": "Multiple fabricated tags",
            "answer": "See [Doc_05 §1] and [Doc_06 §2].",
            "expect_grounded": False,
            "expect_fabricated": 2,
        },
        # --- Edge cases ---
        {
            "name": "No citations and no uncertainty (bad)",
            "answer": "The venue can hold 30 people.",
            "expect_grounded": False,
            "expect_fabricated": 0,
        },
    ]

    passed = 0
    failed = 0
    details = []

    for tc in test_cases:
        report = validate_grounding(tc["answer"], available_tags)

        grounded_ok = report.is_grounded == tc["expect_grounded"]
        fabricated_ok = len(report.fabricated_tags) == tc["expect_fabricated"]
        case_passed = grounded_ok and fabricated_ok

        if case_passed:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"

        details.append({
            "name": tc["name"],
            "status": status,
            "grounded": report.is_grounded,
            "expected_grounded": tc["expect_grounded"],
            "fabricated": report.fabricated_tags,
            "expected_fabricated_count": tc["expect_fabricated"],
            "score": report.score,
        })

    total = passed + failed
    score = (passed / total * 100) if total > 0 else 0

    print("=" * 70)
    print("    GATE 4 — Grounding Validation (G4)")
    print("=" * 70)
    print(f"\n📊 G4 Score: {score:.1f}% ({passed}/{total}) — Target: ≥ 85%")
    print(f"{'─' * 70}")

    for d in details:
        print(f"  {d['status']} {d['name']}")
        print(f"     Grounded: {d['grounded']} (expected {d['expected_grounded']})")
        print(f"     Fabricated: {d['fabricated']} (expected count {d['expected_fabricated_count']})")
        print(f"     Score: {d['score']:.2f}")
        print()

    gate_passed = score >= 85 and all(
        d["expected_fabricated_count"] == 0 or len(d["fabricated"]) > 0
        for d in details
        if d["expected_fabricated_count"] > 0
    )

    print(f"{'=' * 70}")
    if gate_passed:
        print("🟢 GATE 4 PASSED")
    else:
        print("🔴 GATE 4 FAILED")
    print(f"{'=' * 70}")

    return gate_passed


if __name__ == "__main__":
    evaluate_g4()
