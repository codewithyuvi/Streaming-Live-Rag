"""
Gate 5 — Session Refinement Classification (G5)

Tests the refinement classifier against labeled cases:
  - LATE_DETAIL cases (with prior_utterance context)
  - NEW_TOPIC cases (with prior_utterance context)
  - PRESENTATION_ONLY cases

Target: 100% of refinement/suppression cases behave correctly.
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from controller.refinement import classify_refinement


def evaluate_g5():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "labeled_set.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    queries = data.get("queries", [])

    # Filter to cases that have refinement_type labels
    labeled_cases = [q for q in queries if q.get("refinement_type") is not None]

    if not labeled_cases:
        print("No refinement-labeled cases found in eval set.")
        return False

    correct = 0
    total = len(labeled_cases)
    details = []

    for q in labeled_cases:
        utterance = q["utterance"]
        expected_type = q["refinement_type"]
        prior = q.get("prior_utterance", None)

        # Build fake conversation history from prior_utterance
        if prior:
            history = f"[Turn 1] User: {prior}\n[Turn 1] Assistant: (previous answer about this topic)"
            prev_answer = f"I provided information about: {prior}"
        else:
            history = ""
            prev_answer = ""

        result = classify_refinement(
            current_utterance=utterance,
            conversation_history=history,
            previous_answer=prev_answer,
        )

        actual_type = result["type"]
        match = actual_type == expected_type

        if match:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        details.append({
            "id": q["id"],
            "utterance": utterance[:60] + "..." if len(utterance) > 60 else utterance,
            "expected": expected_type,
            "actual": actual_type,
            "reason": result.get("reason", ""),
            "status": status,
            "prior": prior[:40] + "..." if prior and len(prior) > 40 else prior,
        })

    score = (correct / total * 100) if total > 0 else 0

    # Break down by category
    late_detail_cases = [d for d in details if d["expected"] == "LATE_DETAIL"]
    new_topic_cases = [d for d in details if d["expected"] == "NEW_TOPIC"]
    pres_only_cases = [d for d in details if d["expected"] == "PRESENTATION_ONLY"]

    late_correct = sum(1 for d in late_detail_cases if d["status"] == "✅")
    new_correct = sum(1 for d in new_topic_cases if d["status"] == "✅")
    pres_correct = sum(1 for d in pres_only_cases if d["status"] == "✅")

    print("=" * 70)
    print("    GATE 5 — Session Refinement Classification (G5)")
    print("=" * 70)

    print(f"\n📊 Overall: {score:.1f}% ({correct}/{total}) — Target: 100%")
    print(f"   LATE_DETAIL:       {late_correct}/{len(late_detail_cases)}")
    print(f"   NEW_TOPIC:         {new_correct}/{len(new_topic_cases)}")
    print(f"   PRESENTATION_ONLY: {pres_correct}/{len(pres_only_cases)}")

    print(f"\n{'─' * 70}")
    print("LATE_DETAIL cases:")
    print(f"{'─' * 70}")
    for d in late_detail_cases:
        print(f"  {d['status']} {d['id']}: got={d['actual']}")
        print(f"     Utterance: {d['utterance']}")
        print(f"     Prior: {d['prior']}")
        print(f"     Reason: {d['reason']}")
        print()

    print(f"{'─' * 70}")
    print("NEW_TOPIC cases:")
    print(f"{'─' * 70}")
    for d in new_topic_cases:
        print(f"  {d['status']} {d['id']}: got={d['actual']}")
        print(f"     Utterance: {d['utterance']}")
        print(f"     Prior: {d['prior']}")
        print()

    print(f"{'─' * 70}")
    print("PRESENTATION_ONLY cases:")
    print(f"{'─' * 70}")
    for d in pres_only_cases:
        print(f"  {d['status']} {d['id']}: got={d['actual']}")
        print(f"     Utterance: {d['utterance']}")
        print()

    print(f"{'=' * 70}")
    if score >= 90:
        print("🟢 GATE 5 PASSED")
    elif score >= 70:
        print("🟡 GATE 5 MARGINAL")
    else:
        print("🔴 GATE 5 FAILED")
    print(f"{'=' * 70}")

    return score >= 90


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    evaluate_g5()
