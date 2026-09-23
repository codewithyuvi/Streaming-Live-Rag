"""
eval/report.py — Formats and prints the final Judge's Benchmark Scorecard (G1 to G6).
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_scorecard(results: dict):
    print("\n" + "=" * 80)
    print("        SAMSUNG PRISM GenAI HACKATHON 2026–27 — THEME 4")
    print("             STREAMING LIVE RAG BENCHMARK SCORECARD")
    print("=" * 80)
    print(f"{'Gate':<8} | {'Description':<32} | {'Target':<14} | {'Measured':<12} | {'Status'}")
    print("-" * 80)

    gates_meta = [
        ("G1", "Reproducibility & Packaging", "100% clean boot", "g1"),
        ("G2", "Early Retrieval Trigger", ">= 80% eligible", "g2"),
        ("G3", "Multi-Intent Decomposition", ">= 70% compound", "g3"),
        ("G4", "Grounding Support & 0 Fabricated", ">= 85%, 0 fab", "g4"),
        ("G5", "Session Refinement & Suppression", "100% accurate", "g5"),
        ("G6", "Telemetry Observability", "100% coverage", "g6"),
    ]

    all_passed = True
    for gid, desc, target, key in gates_meta:
        gate_res = results.get(key, {})
        status = "PASSED" if gate_res.get("passed", False) else "FAILED"
        status_icon = "🟢 PASS" if gate_res.get("passed", False) else "🔴 FAIL"
        measured = gate_res.get("measured", "N/A")
        if not gate_res.get("passed", False):
            all_passed = False
        print(f"{gid:<8} | {desc:<32} | {target:<14} | {measured:<12} | {status_icon}")

    print("=" * 80)
    if all_passed:
        print("🏆 OVERALL RESULT: ALL 6 COMPETITION GATES PASSED")
    else:
        print("⚠️ OVERALL RESULT: SOME GATES REQUIRE LIVE SERVICE ATTENTION")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__), "results", "scorecard.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print_scorecard(data)
    else:
        print("No scorecard.json found. Run eval/run_eval.py first.")
