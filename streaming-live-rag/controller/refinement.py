"""
Phase 5 — Refinement Classifier

Classifies each new turn relative to the session history into one of:
  - NEW_TOPIC:          Entirely new question unrelated to prior turns
  - LATE_DETAIL:        Adds a constraint or clarification to the previous answer
  - PRESENTATION_ONLY:  No information need (filler, acknowledgement, repetition)

Uses Groq (fast LLM) because this runs in the hot path alongside
the controller and decomposer.

For LATE_DETAIL, the system should refine the previous answer rather
than restart from scratch. For PRESENTATION_ONLY, no retrieval or
synthesis is needed.
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
    from llm_config import call_fast, FAST_LLM_MODEL
except ImportError:
    from ..llm_config import call_fast, FAST_LLM_MODEL


def classify_refinement(
    current_utterance: str,
    conversation_history: str,
    previous_answer: str = "",
) -> dict:
    """
    Classifies the current utterance in the context of conversation history.

    Returns a dict with:
      - "type": "NEW_TOPIC" | "LATE_DETAIL" | "PRESENTATION_ONLY"
      - "reason": short explanation
      - "constraint": (only for LATE_DETAIL) the new constraint to apply
    """
    u_lower = current_utterance.strip().lower()
    
    # Fast path: If no conversation history, check for greetings/chit-chat vs new question
    if not conversation_history.strip() and not previous_answer.strip():
        # Check if it's a greeting/pleasantry
        greeting_words = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "thanks", "thank you"}
        cleaned = re.sub(r"[^\w\s]", "", u_lower)
        if cleaned in greeting_words or any(cleaned.startswith(g) for g in ["hello", "hi ", "hey "]):
            return {
                "type": "PRESENTATION_ONLY",
                "reason": "Initial greeting / conversational opener",
                "constraint": ""
            }
        # If no history and not a greeting, it must be a NEW_TOPIC
        return {
            "type": "NEW_TOPIC",
            "reason": "First substantive question in conversation",
            "constraint": ""
        }

    system_prompt = """You are a conversation turn classifier for a retrieval system.
Given the conversation history and the user's current utterance, classify it into exactly one category.

Output strictly in JSON format:
{
    "type": "NEW_TOPIC" | "LATE_DETAIL" | "PRESENTATION_ONLY",
    "reason": "short explanation",
    "constraint": "the new constraint or detail (only for LATE_DETAIL, empty string otherwise)"
}

Categories:
1. NEW_TOPIC — The user is asking about something completely unrelated to the prior conversation, or a different aspect of a topic requiring new search.
   Examples: Switching from library opening hours to gym membership policies.

2. LATE_DETAIL — The user is modifying, constraining, or clarifying their immediate prior question.
   Examples: "Actually, make that an annual membership", "What about for weekend sessions?", "Does that apply if I join in December?".
   The key signal is that the new utterance MODIFIES or EXTENDS the specific parameters of the previous query.

3. PRESENTATION_ONLY — The user is not asking for new corpus search. This includes:
   - Presentation/format adjustments: reformat, shorten, summarise, bullet points, repeat, say it simpler, translate, change tone.
   - Acknowledgements ("OK", "thanks", "got it", "understood")
   - Filler ("um", "let me think")
   - Exact repetitions or social pleasantries ("great", "perfect", "hello")

Rules:
- If the user asks to reformat, shorten, summarize, or put the prior answer in bullets, ALWAYS classify as PRESENTATION_ONLY.
- A question about a DIFFERENT aspect of the topic is NEW_TOPIC.
- Only classify as LATE_DETAIL when the utterance directly modifies/constrains the parameters of the previous query.
"""

    user_message = f"""Conversation History:
{conversation_history if conversation_history else "(No prior turns)"}

Previous Answer:
{previous_answer if previous_answer else "(No previous answer)"}

Current Utterance:
{current_utterance}"""

    try:
        response = call_fast(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as e:
        return {"type": "NEW_TOPIC", "reason": f"Fallback due to API error: {str(e)}", "constraint": "", "degraded": True}

    try:
        result = json.loads(response.choices[0].message.content)
        valid_types = ["NEW_TOPIC", "LATE_DETAIL", "PRESENTATION_ONLY"]
        if result.get("type") not in valid_types:
            result["type"] = "NEW_TOPIC"
        if "constraint" not in result:
            result["constraint"] = ""
        if "reason" not in result:
            result["reason"] = ""
        return result
    except Exception:
        return {"type": "NEW_TOPIC", "reason": "Parse fallback", "constraint": "", "degraded": True}
