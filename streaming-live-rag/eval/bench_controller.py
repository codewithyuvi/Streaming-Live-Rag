import os
import yaml
from streaming.stream_simulator import simulate_stream
from controller.heuristics import is_stable_enough
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
    
    for q in queries:
        utterance = q["utterance"]
        expected_doc = q.get("expected_doc")
        
        chunks = list(simulate_stream(utterance, words_per_chunk=2))
        
        triggered = False
        triggered_early = False
        
        for i, chunk in enumerate(chunks):
            if not is_stable_enough(chunk.partial_text):
                continue
                
            decision = decide_retrieval(chunk.partial_text)
            
            if decision["trigger"] == "retrieve_now":
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
    
    print("=== Controller Benchmark (G2) ===")
    print(f"Early Retrieval Rate: {g2_score:.1f}% ({early_trigger_count}/{eligible_count})")
    print(f"False-Trigger Rate (Chit-chat): {false_trigger_rate:.1f}% ({false_trigger_count}/{no_retrieval_count})")

if __name__ == "__main__":
    evaluate_controller()
