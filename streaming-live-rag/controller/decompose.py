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
import sys
import json
import re

# Ensure parent directory is on sys.path for llm_config import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from llm_config import call_fast
except ImportError:
    from ..llm_config import call_fast

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
- "What are the library opening hours?" → 1 sub-query
- "What are the library opening hours and how many books can I borrow?" → 2 sub-queries
- "Hello, how are you today?" → 0 sub-queries (empty list)
- "I need the gym membership fee, the pool schedule, and the guest parking policy" → 3 sub-queries
"""

    response = None
    try:
        response = call_fast(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": utterance}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=600
        )
    except Exception as e:
        # If Groq server-side json validation fails or limits tokens, retry cleanly
        try:
            response = call_fast(
                messages=[
                    {"role": "system", "content": system_prompt + "\nOutput strictly valid JSON with no markdown formatting."},
                    {"role": "user", "content": utterance}
                ],
                temperature=0.0,
                max_tokens=600
            )
        except Exception as e2:
            return [{"sub_query": utterance, "intent": "fallback_single", "degraded": True, "error": str(e2)}]

    try:
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
        try:
            result = json.loads(raw_text)
        except Exception:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
            else:
                raise
        sub_queries = result.get("sub_queries", [])

        # Validate and enforce cap
        if not isinstance(sub_queries, list):
            return [{"sub_query": utterance, "intent": "parse_fallback"}]

        # Enforce hard cap
        sub_queries = sub_queries[:MAX_SUB_QUERIES]

        # Ensure each item has required non-empty string fields
        validated = []
        for sq in sub_queries:
            if isinstance(sq, dict) and isinstance(sq.get("sub_query"), str) and sq["sub_query"].strip():
                validated.append({
                    "sub_query": sq["sub_query"].strip(),
                    "intent": str(sq.get("intent", "sub_intent")).strip()
                })

        # If decomposer explicitly returned an empty list (Rule 6: chit-chat/greeting),
        # respect it rather than forcing a fallback search query
        if not validated and isinstance(sub_queries, list) and len(sub_queries) == 0:
            q_words = {"what", "when", "where", "who", "why", "how", "which"}
            first_words = set(utterance.lower().split()[:3])
            if not (first_words & q_words):
                return []
            return [{"sub_query": utterance, "intent": "single"}]

        if not validated and len(utterance.split()) > 2:
            return [{"sub_query": utterance, "intent": "single"}]

        return validated

    except Exception:
        return [{"sub_query": utterance, "intent": "parse_fallback"}]
