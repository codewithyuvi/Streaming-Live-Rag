# Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Official corpus arrives late or differs structurally from assumptions** | Medium | High | Placeholder corpus + corpus-agnostic ingestion; re-run the full eval set within hours of arrival; Phase 7–8 carry deliberate buffer. |
| **LLM API rate limits/cost during heavy benchmark replay** | Medium | Medium | Cheap/fast model for controller + decomposition, stronger model reserved for synthesis only; cache LLM calls during dev; exponential backoff. |
| **Controller misses the G2 80% threshold** | Medium-High | High | Full dedicated day (Phase 3); numeric gate before any downstream work starts; thresholds kept in config, not hardcoded, so they're fast to retune. |
| **Decomposer over- or under-fragments** | Medium | Medium | Explicit single-question control cases; hard cap on sub-query count; few-shot guardrails against Pitfall 5. |
| **Citation hallucination** | Medium | High | Deterministic ID-membership validator, not "the LLM said it was fine" (ADR-5). |
| **Session bugs — state bleeding across sessions, or not persisting within one** | Low-Medium | High | Minimal, directly testable session store; explicit isolation unit tests. |
| **Docker fails on a judge's clean machine** | Medium | High (G1 is binary) | Test on a genuinely clean environment starting Phase 6, not only dev laptops; pin every dependency version. |
| **Team time crunch (exams, other commitments across 9 days)** | Medium | Medium | Strict daily gate discipline; MVP-first scope; parallel workstreams from Day 1. |
| **Held-out benchmark structurally different from the guide's 3 worked examples** | Low-Medium | Medium | Keep logic general and config-driven — never pattern-matched to the guide's own examples. |
| **Reranker adds latency without a quality gain on your actual corpus** | Medium | Low-Medium | Ablate reranker on/off on your own eval set (Phase 2) before assuming it helps. |
| **Judge's machine lacks your LLM API key** | Low-Medium | High (blocks G1) | Document required env vars precisely; default to a provider with a fast, free signup path (Groq) so obtaining a key is a 2-minute step. |
