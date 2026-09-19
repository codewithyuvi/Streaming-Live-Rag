"""
Phase 4 — Multi-Intent Decomposer

Uses Groq (fast LLM) to split compound user utterances into orthogonal
sub-queries. Hard-capped at MAX_SUB_QUERIES to prevent over-fragmentation.

Design decisions:
- Groq is used instead of Gemini because decomposition runs in the hot path
  and must be fast (<500ms).
- Single-intent queries return a 1-element list (no fragmentation).
- JSON mode ensures structured output.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")

MAX_SUB_QUERIES = 4


def decompose_query(utterance: str) -> list[dict]:
    """
    Decomposes a user utterance into orthogonal sub-queries.

    Returns a list of dicts, each with:
      - "sub_query": str  — the rewritten sub-question
      - "intent":    str  — short label for the intent

    For single-intent utterances, returns a single-element list.
    Hard-capped at MAX_SUB_QUERIES items.
    """

    system_prompt = f"""You are a query decomposer for a retrieval system.
Given a user utterance, determine if it contains MULTIPLE DISTINCT information needs.

Rules:
1. If the utterance asks about ONE thing, return exactly ONE sub-query.
2. If the utterance asks about MULTIPLE DISTINCT topics, split into separate sub-queries.
3. Maximum {MAX_SUB_QUERIES} sub-queries.
4. Each sub-query must be a self-contained, searchable question.
5. Do NOT split a single question into redundant paraphrases.
6. If the utterance is a greeting or chit-chat with no information need, return an empty list.

Output strictly in JSON format:
{{
    "sub_queries": [
        {{"sub_query": "rewritten searchable question", "intent": "short_label"}},
        ...
    ]
}}

Examples:
- "What is the capacity?" → 1 sub-query
- "What is the capacity and who approves flights?" → 2 sub-queries
- "Hello, how are you?" → 0 sub-queries (empty list)
- "I need the cancellation policy, the travel approval process, and the venue capacity" → 3 sub-queries
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=FAST_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": utterance}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            # Fallback: treat the whole utterance as a single query
            return [{"sub_query": utterance, "intent": "fallback_single"}]

    try:
        result = json.loads(response.choices[0].message.content)
        sub_queries = result.get("sub_queries", [])

        # Validate and enforce cap
        if not isinstance(sub_queries, list):
            return [{"sub_query": utterance, "intent": "parse_fallback"}]

        # Enforce hard cap
        sub_queries = sub_queries[:MAX_SUB_QUERIES]

        # Ensure each item has required fields
        validated = []
        for sq in sub_queries:
            if isinstance(sq, dict) and "sub_query" in sq:
                validated.append({
                    "sub_query": sq["sub_query"],
                    "intent": sq.get("intent", "unknown")
                })

        # If decomposer returned nothing but there's real content, use original
        if not validated and len(utterance.split()) > 2:
            return [{"sub_query": utterance, "intent": "single"}]

        return validated

    except Exception:
        return [{"sub_query": utterance, "intent": "parse_fallback"}]
