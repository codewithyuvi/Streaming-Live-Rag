# ADR-4: LLM Provider — Dual-Provider Architecture (Groq + Gemini)

**Decision:** We will use a **Dual-Provider Architecture**.
1. **Groq (`llama3-8b-8192`)** will handle all latency-sensitive operations (the Streaming Controller deciding Wait/Retrieve/Suppress, and the Multi-Intent Decomposer).
2. **Gemini (`gemini-3.8-pro`)** will handle the final Session-Aware Synthesis.

**Why:**
The hackathon theme strictly grades two conflicting dimensions: **Time-to-First-Token/Latency** and **Reasoning Quality**. 
During Day 1 POC testing, we attempted to use Gemini for the controller tasks. While it produced excellent reasoning, the latency averaged 5,000ms - 16,000ms. Because the Streaming Controller must evaluate incoming chunks multiple times per second, this latency would completely freeze the streaming pipeline and fail the "Early Retrieval" gates.

Groq's LPU-based hosting offers token generation speeds an order of magnitude faster than typical cloud endpoints, reliably returning routing decisions in ~200ms-400ms. However, `llama3-8b` may struggle with the complex citation and grounding constraints required for the final answer. Therefore, we offload the heavy synthesis to Gemini, which only runs once per retrieval event rather than continuously on every chunk.

**Alternatives considered:** 
- *Gemini Only:* Rejected due to 5s+ latency causing pipeline freezes.
- *Ollama Cloud API:* Rejected because free-tier Ollama clouds do not guarantee the ultra-low latency of LPUs, adding unnecessary risk to the G2 Early Retrieval gate.
- *Groq Only:* Plausible fallback, but larger models like `llama3-70b` on Groq have stricter rate limits on the free tier, making heavy benchmark testing risky.

**Revisit if:** Groq rate limits block automated evaluation scripts on Day 7, requiring us to swap the fast provider to Cerebras.
