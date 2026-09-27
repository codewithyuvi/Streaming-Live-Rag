"""
scripts/live_chat.py — Interactive Live CLI for Testing Custom Queries & Data.

Allows interactive testing against the running FastAPI pipeline server
with custom questions, multi-turn follow-ups, and live inspection of
citations, answer versioning, and grounding scores.
"""

import sys
import time
import requests

SERVER_URL = "http://localhost:8000"


def main():
    print("=" * 70)
    print("      STREAMING LIVE RAG — INTERACTIVE LIVE TEST CONSOLE")
    print("=" * 70)
    print("Commands:")
    print("  'new' or 'reset'  -> Start a fresh conversation session")
    print("  'exit' or 'quit'   -> Exit the console\n")

    # Check if server is running
    try:
        res = requests.get(f"{SERVER_URL}/health", timeout=3)
        if res.status_code != 200:
            print(f"[!] Server at {SERVER_URL} returned status {res.status_code}.")
            return
        print(f"[+] Connected to Streaming Live RAG API at {SERVER_URL}")
    except Exception:
        print(f"[!] Could not connect to API server at {SERVER_URL}.")
        print("    Please ensure the server is running:")
        print("    1. Start Docker Desktop and run: docker compose up")
        print("       OR")
        print("    2. Start API locally: python -m uvicorn api.main:app --port 8000")
        return

    session_id = f"live_{int(time.time())}"
    turn_id = 1
    print(f"\n[Session initialized: {session_id}]")

    while True:
        try:
            print("-" * 70)
            user_input = input(f"[Turn {turn_id}] Enter query > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Exiting live console. Goodbye!")
                break

            if user_input.lower() in ("new", "reset"):
                session_id = f"live_{int(time.time())}"
                turn_id = 1
                print(f"\n[+] New session started: {session_id}\n")
                continue

            print("Thinking and retrieving...")
            t0 = time.time()
            resp = requests.post(
                f"{SERVER_URL}/turn",
                json={
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "utterance": user_input,
                },
                timeout=90,
            )

            if resp.status_code != 200:
                print(f"[!] Error ({resp.status_code}): {resp.text}")
                continue

            data = resp.json()
            ans = data.get("answer", "")
            telem = data.get("telemetry", {})

            ref_type = telem.get("refinement_type", "UNKNOWN")
            version = telem.get("answer_version", 1)
            citations = telem.get("citations", [])
            grounding = telem.get("grounding_score", 0.0)
            sub_queries = telem.get("sub_queries", [])
            latencies = telem.get("latencies_ms", {})
            e2e = latencies.get("end_to_end", round((time.time() - t0) * 1000, 1))

            print("\n" + "=" * 50)
            print("ANSWER:")
            print("=" * 50)
            print(ans)
            print("=" * 50)
            print("METADATA & OBSERVABILITY:")
            print(f"  • Refinement Type: {ref_type}")
            print(f"  • Answer Version:  v{version}")
            print(f"  • Cited Tags:      {', '.join(citations) if citations else 'None'}")
            print(f"  • Grounding Score: {grounding * 100:.1f}%")
            if sub_queries:
                print(f"  • Decomposed Sub-queries ({len(sub_queries)}):")
                for sq in sub_queries:
                    print(f"      - {sq}")
            print(f"  • End-to-End Latency: {e2e} ms")

            if telem.get("uncertainty"):
                print(f"  [!] Uncertainty Flag: {telem['uncertainty']}")

            turn_id += 1

        except (KeyboardInterrupt, EOFError):
            print("\nExiting live console. Goodbye!")
            break
        except Exception as e:
            print(f"[!] Request error: {e}")


if __name__ == "__main__":
    main()
