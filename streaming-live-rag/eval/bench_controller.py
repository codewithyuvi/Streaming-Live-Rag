import os
import sys
import time
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from streaming.stream_simulator import simulate_stream
from controller.heuristics import is_stable_enough, get_stable_query_prefix
from controller.decide import decide_retrieval

def evaluate_controller():
    yaml_path = os.path.join(os.path.dirname(__file__), "labeled_set.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    queries = data.get("queries", [])
    
    eligible_count = 0
    early_trigger_count = 0
    
    no_retrieval_count = 0
    false_trigger_count = 0
    latencies = []
    
    for q in queries:
        utterance = q["utterance"]
        expected_doc = q.get("expected_doc")
        
        chunks = list(simulate_stream(utterance, words_per_chunk=2))
        
        triggered = False
        triggered_early = False
        
        for i, chunk in enumerate(chunks):
            candidate = get_stable_query_prefix(chunk.partial_text)
            if not candidate:
                continue
                
            t0 = time.perf_counter()
            decision = decide_retrieval(candidate)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            
            if decision.get("trigger") == "retrieve_now":
                triggered = True
                if i < len(chunks) - 1:
                    triggered_early = True
                break
                
        if expected_doc is None and "capacity" not in utterance.lower(): # Basic filter for chit-chat vs unanswerable
            no_retrieval_count += 1
            if triggered:
                false_trigger_count += 1
                
        elif expected_doc is not None:
            # Eligible for early retrieval
            if len(chunks) > 2: # Needs to be long enough to even trigger early
                eligible_count += 1
                if triggered_early:
                    early_trigger_count += 1

    g2_score = (early_trigger_count / eligible_count) * 100 if eligible_count > 0 else 0
    false_trigger_rate = (false_trigger_count / no_retrieval_count) * 100 if no_retrieval_count > 0 else 0

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lats = sorted(latencies)
    p50_lat = sorted_lats[int(len(sorted_lats) * 0.5)] if sorted_lats else 0.0
    p95_lat = sorted_lats[min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)] if sorted_lats else 0.0
    
    print("=== Controller Benchmark (G2) ===")
    print(f"Early Retrieval Rate: {g2_score:.1f}% ({early_trigger_count}/{eligible_count})")
    print(f"False-Trigger Rate (Chit-chat): {false_trigger_rate:.1f}% ({false_trigger_count}/{no_retrieval_count})")
    print(f"Decision Latency: Avg={avg_lat:.1f}ms, P50={p50_lat:.1f}ms, P95={p95_lat:.1f}ms (N={len(latencies)})")

if __name__ == "__main__":
    evaluate_controller()
