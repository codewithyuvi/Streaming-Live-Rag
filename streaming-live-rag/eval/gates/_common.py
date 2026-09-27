"""
eval/gates/_common.py — shared helpers for the G1..G6 gate scripts.

Every gate:
  * is runnable standalone:  python eval/gates/gN_*.py
  * prints PASS / FAIL / SKIP with numbers
  * exits 0 (pass), 1 (fail), 2 (skipped)

Skips are honest: a gate that needs a live LLM key raises GateSkipped with a
clear "needs <KEY>" message instead of producing a fake pass or fail.
"""

import os
import sys
import traceback

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


class GateSkipped(Exception):
    """Raised when a gate cannot run because a required live service is absent."""


def require_env(key: str) -> str:
    """Return the env var value, or raise GateSkipped with a clear message."""
    val = os.getenv(key)
    if not val:
        raise GateSkipped(f"needs {key}: set {key} in the environment to run this gate")
    return val


def gate_main(gate_id: str, evaluate_fn) -> int:
    """
    Run evaluate_fn() -> (status, metric_str), print the verdict, and return
    the process exit code: 0 = pass, 1 = fail, 2 = skipped.
    """
    try:
        status, metric = evaluate_fn()
    except GateSkipped as e:
        print(f"\n[{gate_id}] SKIP — {e}")
        print(f"[{gate_id}] Set the required key(s) and re-run for a real result.")
        return 2
    except Exception:
        print(f"\n[{gate_id}] FAIL — unexpected error:")
        traceback.print_exc()
        return 1

    if status == STATUS_PASS:
        print(f"\n[{gate_id}] PASS — {metric}")
        return 0
    if status == STATUS_SKIP:
        print(f"\n[{gate_id}] SKIP — {metric}")
        return 2
    print(f"\n[{gate_id}] FAIL — {metric}")
    return 1


def banner(title: str) -> None:
    print("=" * 70)
    print(f"    {title}")
    print("=" * 70)


def retrieval_preflight() -> None:
    """Grounding/refinement gates are unmeasurable without a live retrieval
    backend. Raises GateSkipped (not a failure) when Qdrant or the indexed
    corpus is unavailable."""
    try:
        from retrieval.ingest import get_corpus_summary
    except Exception as e:
        raise GateSkipped(f"needs Qdrant + indexed corpus (import failed: {e})")
    try:
        summary = get_corpus_summary()
    except Exception as e:
        raise GateSkipped(f"needs Qdrant + indexed corpus (unreachable: {e})")
    if summary.get("total_chunks", 0) < 4:
        raise GateSkipped(
            "needs Qdrant + indexed corpus "
            f"(only {summary.get('total_chunks', 0)} chunks indexed)")


def footer(passed: bool, label: str) -> None:
    print("=" * 70)
    print(f"{'PASS' if passed else 'FAIL'} — {label}")
    print("=" * 70)
