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
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")


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

    system_prompt = """You are a conversation turn classifier for a retrieval system.
Given the conversation history and the user's current utterance, classify it into exactly one category.

Output strictly in JSON format:
{
    "type": "NEW_TOPIC" | "LATE_DETAIL" | "PRESENTATION_ONLY",
    "reason": "short explanation",
    "constraint": "the new constraint or detail (only for LATE_DETAIL, empty string otherwise)"
}

Categories:
1. NEW_TOPIC — The user is asking about something completely unrelated to the prior conversation.
   Examples: switching from venue questions to travel questions with no connection.

2. LATE_DETAIL — The user is adding a constraint, clarification, or follow-up to their previous question.
   Examples: "Actually, the trip was international", "What about for groups larger than 20?",
   "And what's the penalty if I cancel?", "I meant for the Pune venue specifically".
   The key signal is that the new utterance MODIFIES or EXTENDS the previous topic.

3. PRESENTATION_ONLY — The user is not asking for information. This includes:
   - Acknowledgements ("OK", "thanks", "got it")
   - Filler ("um", "let me think")
   - Exact repetitions of what was just said
   - Social pleasantries mid-conversation ("great", "perfect")

Rules:
- If there is NO conversation history, the answer is always NEW_TOPIC.
- A question about a DIFFERENT aspect of the SAME document/topic is still NEW_TOPIC.
- Only classify as LATE_DETAIL when the utterance directly modifies/constrains the previous query.
"""

    user_message = f"""Conversation History:
{conversation_history if conversation_history else "(No prior turns)"}

Previous Answer:
{previous_answer[:300] if previous_answer else "(No previous answer)"}

Current Utterance:
{current_utterance}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=FAST_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            return {"type": "NEW_TOPIC", "reason": f"Fallback due to API error: {str(e)}", "constraint": ""}

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
        return {"type": "NEW_TOPIC", "reason": "Parse fallback", "constraint": ""}
