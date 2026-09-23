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

    # Write Markdown Scorecard (L6)
    try:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        md_path = os.path.join(results_dir, "scorecard.md")
        lines = [
            "# Samsung PRISM GenAI Hackathon 2026–27 — Theme 4",
            "## Streaming Live RAG Benchmark Scorecard\n",
            "| Gate | Description | Target | Measured | Status |",
            "| :--- | :--- | :--- | :--- | :---: |",
        ]
        for gid, desc, target, key in gates_meta:
            gate_res = results.get(key, {})
            status_icon = "🟢 PASS" if gate_res.get("passed", False) else "🔴 FAIL"
            measured = gate_res.get("measured", "N/A")
            lines.append(f"| **{gid}** | {desc} | {target} | {measured} | {status_icon} |")
        lines.append(f"\n**Overall Result:** {'🏆 ALL 6 COMPETITION GATES PASSED' if all_passed else '⚠️ SOME GATES REQUIRE ATTENTION'}\n")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__), "results", "scorecard.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print_scorecard(data)
    else:
        print("No scorecard.json found. Run eval/run_eval.py first.")
