"""
eval/run_eval.py — Master Benchmark & Gate Evaluation Harness (G1 to G6).

Runs each gate's evaluate_gN() -> (status, metric). Statuses: pass / fail / skip.
A gate raises GateSkipped when its live requirements (API keys, Qdrant) are
absent — that is recorded as "skip", never a fake pass/fail.

Scorecard guard: if EVERY gate was skipped (fully offline), the existing
scorecard is left untouched — we do not overwrite it with an all-skipped run.

Exit codes: 0 = all gates passed; 1 = any gate failed; 2 = some skipped, none failed.
"""

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

from eval.gates._common import GateSkipped, STATUS_PASS, STATUS_FAIL, STATUS_SKIP
from eval.gates.g1_reproducibility import evaluate_g1
from eval.gates.g2_early_retrieval import evaluate_g2
from eval.gates.g3_multi_intent import evaluate_g3
from eval.gates.g4_grounding import evaluate_g4
from eval.gates.g5_session_refinement import evaluate_g5
from eval.gates.g6_telemetry import evaluate_g6
from eval.report import print_scorecard

GATES = [
    ("g1", "Reproducibility & Packaging", "100% checks", evaluate_g1),
    ("g2", "Early Retrieval Trigger", ">= 80% eligible", evaluate_g2),
    ("g3", "Multi-Intent Decomposition", ">= 70% compound", evaluate_g3),
    ("g4", "Grounding Support & 0 Fabricated", ">= 85%, 0 fab", evaluate_g4),
    ("g5", "Session Refinement & Suppression", "verified state continuity", evaluate_g5),
    ("g6", "Telemetry Observability", "100% trace coverage", evaluate_g6),
]


def run_full_eval():
    start_time = time.time()
    results = {}

    print("\n" + "=" * 80)
    print("      STARTING FULL BENCHMARK EVALUATION RUN (GATES G1 TO G6)")
    print("=" * 80 + "\n")

    for gid, name, target, fn in GATES:
        try:
            status, metric = fn()
        except GateSkipped as e:
            status, metric = STATUS_SKIP, f"Skipped: {e}"
            print(f"\n[{gid.upper()}] SKIP — {e}\n")
        except Exception as e:  # unexpected error inside a gate = fail with reason
            import traceback
            traceback.print_exc()
            status, metric = STATUS_FAIL, f"Error: {e}"
        results[gid] = {"name": name, "passed": status == STATUS_PASS,
                        "status": status, "measured": metric, "target": target}

    # Guard: don't overwrite the scorecard when nothing actually ran.
    ran = [g for g in results.values() if g["status"] != STATUS_SKIP]
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    scorecard_path = os.path.join(results_dir, "scorecard.json")
    if ran:
        with open(scorecard_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results written to: {scorecard_path}")
    else:
        print("All gates skipped (offline) — existing scorecard left untouched.")

    print_scorecard(results)
    print(f"Total evaluation time: {time.time() - start_time:.2f}s\n")

    if any(g["status"] == STATUS_FAIL for g in results.values()):
        return 1
    if any(g["status"] == STATUS_SKIP for g in results.values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(run_full_eval())
