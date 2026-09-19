# ADR-4: LLM provider — pluggable via env var, Gemini API chosen for balance of speed and reasoning

**Decision:** All LLM calls (controller classification, decomposition, synthesis) go through one thin client interface selected by an `LLM_PROVIDER` env var. Default recommendation: Gemini API (via Google AI Studio). Specifically, `gemini-1.5-flash` for the latency-sensitive controller and decomposition calls, and `gemini-1.5-pro` for final synthesis, where answer quality matters most.

**Why:** Two of the theme's own graded reporting dimensions are **time-to-first-token** and **cost per turn** — provider choice directly moves both numbers. Gemini 1.5 Flash provides exceptional time-to-first-token latency (often faster than comparable open models) and very low cost, making it ideal for the high-volume streaming controller that evaluates text chunks multiple times a second. Additionally, staying within the Gemini ecosystem allows seamless switching to `gemini-1.5-pro` for the final synthesis step without changing client libraries or prompt formatting heavily.

**Caution:** Ensure you are using the `google-genai` SDK rather than the legacy `google-generativeai` package, as the newer SDK has better typing and async support which is critical for our asyncio pipeline.

**Revisit if:** Rate limits become an issue during heavy parallel evaluation testing, though AI Studio's free tiers are generally generous enough for hackathon development loops.
