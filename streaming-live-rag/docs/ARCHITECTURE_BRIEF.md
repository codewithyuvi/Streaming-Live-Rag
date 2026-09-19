# System Architecture Brief

This document outlines the architecture for the Streaming Live RAG system based on the 9-Day Execution Plan.

## 1. Component Flow

1. **Simulated streaming utterance (test harness):** Emits timestamped chunks.
2. **Stream Simulator:** Replays a full utterance as timestamped partial chunks.
3. **Retrieval Controller:** Decides whether to `Wait`, `Retrieve`, or `Suppress`. Logs trigger and reason.
4. **Multi-Intent Decomposer:** If `Retrieve`, splits the utterance into 1..N orthogonal sub-queries.
5. **Hybrid Search (Qdrant):** Performs dense + BM25 sparse search.
6. **RRF Fusion + Dedup:** Merges results.
7. **Cross-Encoder Rerank:** Reranks the merged sub-query results.
8. **Session-Aware Synthesizer:** Generates grounded answers with citations, uncertainty, and versioning. Also performs refinement classification.
9. **Streamed answer + citations:** Final output.

(See diagram in Section 4 of the execution plan PDF).

## 2. Component Boundaries

| Component | Responsible for | Interface |
| :--- | :--- | :--- |
| **Stream Simulator** | Replaying full utterance as timestamped chunks. | `iterate_chunks(utterance) -> Iterator[StreamChunk]` |
| **Retrieval Controller** | Wait/Retrieve/Suppress decision per chunk. | `decide(session, chunks_so_far) -> Decision` |
| **Multi-Intent Decomposer**| Splitting utterance into orthogonal sub-queries. | `decompose(utterance) -> list[str]` |
| **Hybrid Search** | Dense + sparse search and RRF fusion per sub-query. | `search(sub_query) -> list[Chunk]` |
| **Reranker** | Cross-encoder rerank + dedup across merged results. | `rerank(query, chunks) -> list[Chunk]` |
| **Session-Aware Synthesizer**| Grounded answer + citations + uncertainty + versioning. | `synthesize(session, evidence) -> Answer` |
| **Session Store** | Ephemeral per-session turns, citations, answer versions. | `get(session_id)`, `update(...)` |
| **Telemetry Sink** | One structured JSON event per turn. | `emit(event: TelemetryEvent)` |
| **Eval Harness** | Replays the labeled set through live API, computes G1-G6. | `run_eval(set) -> Scorecard` |

## 3. Core Data Model

The system uses a structured telemetry logging schema. 

```json
{
  "session_id": "sess_8f2a",
  "turn_id": 3,
  "controller_decision": {"trigger": "multi_intent", "timestamp_s": 1.6, "reason": "..."},
  "retrieval_events": [
    {"timestamp_s": 0.8, "query": "Pune workshop venue capacity 30", "trigger": "provisional"},
    {"timestamp_s": 1.6, "query": "cancellation policy workshop venues Pune", "trigger": "multi_intent"}
  ],
  "sub_queries": ["venue capacity for 30 attendees in Pune", "cancellation terms and refund policies"],
  "answer": "...",
  "citations": ["Doc_12 §2", "Doc_31 §4"],
  "uncertainty": "Catering accommodation policies... could not be verified.",
  "answer_version": 1,
  "latencies_ms": {"retrieval": 140, "rerank": 90, "time_to_first_token": 310, "end_to_end": 980},
  "token_cost": {"input": 612, "output": 184, "usd_estimate": 0.0007}
}
```

## 4. Final Stack at a Glance

| Layer | Choice | Reason |
| :--- | :--- | :--- |
| **API/runtime** | Python 3.11+, FastAPI, asyncio, uvicorn | Async-native, minimal. |
| **Orchestration** | Hand-rolled pipeline (no agent framework) | ADR-1 |
| **Vector + Sparse DB**| Qdrant (Docker) | ADR-2 |
| **Embeddings/rerank** | FastEmbed (bge-small-en-v1.5 + ms-marco-MiniLM-L-6-v2) | ADR-3 |
| **LLM provider** | Pluggable; Groq default | ADR-4 |
| **Grounding check** | Deterministic ID-membership validator | ADR-5 |
| **Schema/validation** | Pydantic | Free runtime validation. |
| **Testing** | pytest | Unit tests + gate measurement scripts. |
| **Containerization** | Docker + docker-compose | Single-command boot (G1). |
| **Telemetry** | Custom structured JSON logging | Zero external dependency (G6). |
