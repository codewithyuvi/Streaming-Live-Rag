# ADR-1: Orchestration — hand-rolled FastAPI/asyncio, not LangChain/LangGraph/full LlamaIndex

**Decision:** The controller -> decomposer -> retrieval -> synthesis -> refinement pipeline is plain Python (`FastAPI` + `asyncio.gather` for parallelism), not built on top of an agent framework.

**Why:** Head-to-head 2026 benchmarks of LangChain, LangGraph, LlamaIndex, and Haystack running an equivalent pipeline found LangGraph carrying the highest orchestration overhead and token usage of the group, with LangChain close behind, while Haystack and LlamaIndex ran leaner. More importantly, *none* of these frameworks has a first-class primitive for ”decide whether to retrieve from an incomplete, still-arriving utterance” — that's the genuinely novel part of this theme, and you'd be writing custom code around the framework either way. Given the guide's explicit ”Architectural Parsimony” rule and the fact that framework churn (deprecations, breaking API changes) is a real, currently-reported risk in this ecosystem, the lowest-risk path for 9 days is full control over the 5 custom components, wired together with nothing heavier than `asyncio`.

**Alternatives considered:** 
- LlamaIndex as backbone (fastest path to a RAG pipeline, but its chat-engine/agent abstractions don't map to incremental-chunk-driven retrieval, so you'd fight the framework exactly where the theme is hardest). 
- LangGraph (built for stateful multi-step agents, but that's more machinery than ”controller decides retrieve-or-not, then a linear pipeline runs” needs — the guide is explicitly skeptical of this).

**Revisit if:** a team member has deep, current LangGraph experience and the state-machine/checkpointing primitives would demonstrably save days, not add a debugging tax.
