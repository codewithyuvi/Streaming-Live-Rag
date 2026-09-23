"""
Gate 2 — Early Retrieval Evaluation (G2).

Measures:
1. Early retrieval rate: For eligible simple queries, retrieval fires early
   before speech completes (t_retrieval < t_end). Target: >= 80%.
2. False-trigger rate: Conversational chit-chat should NOT trigger retrieval.
   Target: 0% false-trigger.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

from streaming.stream_simulator import simulate_stream
from controller.heuristics import is_stable_enough, get_stable_query_prefix
from controller.decide import decide_retrieval


def evaluate_g2():
    print("=" * 70)
    print("    GATE 2 — Early Retrieval & False-Trigger Rate (G2)")
    print("=" * 70)

    # Test cases: eligible early queries vs chit-chat suppression controls
    eligible_test_cases = [
        {"id": "q01", "utterance": "What is the maximum capacity of the workshop hall?", "expect_early": True},
        {"id": "q02", "utterance": "What is the cancellation policy for venue bookings?", "expect_early": True},
        {"id": "q03", "utterance": "Who needs to approve international travel requests?", "expect_early": True},
        {"id": "q04", "utterance": "What is the reimbursement limit for client dinners?", "expect_early": True},
        {"id": "q05", "utterance": "Where can I find the projector remote in room 3B?", "expect_early": True},
        {"id": "q06", "utterance": "How far in advance must international flights be booked?", "expect_early": True},
        {"id": "q07", "utterance": "What is the policy for weekend team building events?", "expect_early": True},
        {"id": "q08", "utterance": "Can external guests park in the main visitor lot?", "expect_early": True},
    ]

    suppression_test_cases = [
        {"id": "c01", "utterance": "Hello, how are you today?", "expect_suppress": True},
        {"id": "c02", "utterance": "Good morning everyone, can you hear me ok?", "expect_suppress": True},
        {"id": "c03", "utterance": "Let me pull up my presentation slides real quick.", "expect_suppress": True},
        {"id": "c04", "utterance": "Thank you for joining the meeting.", "expect_suppress": True},
    ]

    # Evaluate heuristic stability + direct decide_retrieval streaming timeline
    early_triggered = 0
    total_eligible = len(eligible_test_cases)
    details = []
    latencies = []

    for tc in eligible_test_cases:
        chunks = list(simulate_stream(tc["utterance"], words_per_chunk=2, ms_per_chunk=300))
        t_end = chunks[-1].t_offset_s if chunks else 0.0
        
        stable_chunk = None
        for chunk in chunks[:-1]:
            candidate = get_stable_query_prefix(chunk.partial_text)
            if candidate:
                t0 = time.perf_counter()
                decision = decide_retrieval(candidate)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)

                if decision.get("trigger") == "retrieve_now":
                    stable_chunk = chunk
                    break

        fired_early = stable_chunk is not None and stable_chunk.t_offset_s < t_end
        if fired_early:
            early_triggered += 1

        details.append({
            "id": tc["id"],
            "utterance": tc["utterance"][:50] + "...",
            "t_end": t_end,
            "t_fire": stable_chunk.t_offset_s if stable_chunk else t_end,
            "fired_early": fired_early
        })

    false_triggers = 0
    total_suppressed = len(suppression_test_cases)
    for sc in suppression_test_cases:
        chunks = list(simulate_stream(sc["utterance"], words_per_chunk=2, ms_per_chunk=300))
        triggered = False
        for chunk in chunks:
            candidate = get_stable_query_prefix(chunk.partial_text)
            if candidate:
                t0 = time.perf_counter()
                decision = decide_retrieval(candidate)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)

                if decision.get("trigger") == "retrieve_now":
                    triggered = True
                    break
        if triggered:
            false_triggers += 1

    early_rate = (early_triggered / total_eligible) * 100 if total_eligible > 0 else 0
    false_trigger_rate = (false_triggers / total_suppressed) * 100 if total_suppressed > 0 else 0

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p50_lat = sorted(latencies)[int(len(latencies) * 0.5)] if latencies else 0.0
    p95_lat = sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0

    print(f"\n📊 Early Retrieval Rate: {early_rate:.1f}% ({early_triggered}/{total_eligible}) — Target: >= 80%")
    print(f"📊 False-Trigger Rate:   {false_trigger_rate:.1f}% ({false_triggers}/{total_suppressed}) — Target: 0%")
    print(f"📊 Controller Latency:   Avg={avg_lat:.1f}ms, P50={p50_lat:.1f}ms, P95={p95_lat:.1f}ms (N={len(latencies)})")
    print("─" * 70)
    for d in details:
        status = "⚡ EARLY" if d["fired_early"] else "⏱️ END"
        print(f"  {status} {d['id']}: t_fire={d['t_fire']}s < t_end={d['t_end']}s | {d['utterance']}")

    passed = early_rate >= 80.0 and false_trigger_rate == 0.0

    print("=" * 70)
    if passed:
        print("🟢 GATE 2 PASSED (Early Retrieval & Suppression Enforced)")
    else:
        print("🔴 GATE 2 FAILED")
    print("=" * 70)

    return passed, f"{early_rate:.1f}%"


if __name__ == "__main__":
    evaluate_g2()
