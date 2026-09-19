# ADR-5: Grounding mechanism — deterministic ID-membership check as the primary gate

**Decision:** G4 is measured by a fast, deterministic post-processor: parse every `[Doc_ID §Section]` tag out of the generated answer, and fail/flag any citation whose ID was not actually in that turn's retrieved-chunk set. LLM-judge–style faithfulness scoring (Ragas, DeepEval) is optional and secondary.

**Why:** ”Zero fabricated or hallucinated document IDs” (the guide's own G4 wording) is a verifiable claim — you don't need an LLM's opinion to check whether a cited ID exists in a Python set. This is instant, free, and 100% reproducible, which matters because G4 is a hard pass/fail gate, not a soft quality signal. Ragas/DeepEval remain genuinely useful as a **secondary, reportable** faithfulness metric for the Benchmarking & Evaluation Report (both are actively maintained in 2026, with DeepEval favoring pytest-style CI integration and Ragas being the more RAG-specific, research-native tool) — use one of them to add rigor to the report, not to gate the pipeline.

**Revisit if:** you have spare time on Day 8 and want a second, LLM-graded number for the report's credibility — add it, but never let it replace the deterministic check.
