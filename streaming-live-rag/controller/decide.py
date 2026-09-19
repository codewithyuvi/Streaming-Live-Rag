import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")

def decide_retrieval(partial_utterance: str) -> dict:
    """
    Classifies a partial utterance into one of three states:
    1. "wait" - The user hasn't finished expressing the core entity or question.
    2. "retrieve_now" - The user has provided enough context to search the database.
    3. "no_retrieval_needed" - The user is just chatting or asked a question that doesn't need external data.
    """
    system_prompt = """
    You are a real-time speech controller. You see words as a user speaks them.
    Your job is to decide if we have enough information to search a database yet.
    
    Output strictly in JSON format:
    {
        "trigger": "wait" | "retrieve_now" | "no_retrieval_needed",
        "reason": "short explanation"
    }
    
    Rules:
    - If it's just a greeting ("hello", "how are you"), output "no_retrieval_needed".
    - If the core noun/question isn't finished (e.g. "what is the maximum..."), output "wait".
    - If a distinct, searchable concept is formed (e.g. "what is the maximum capacity?"), output "retrieve_now".
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=FAST_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": partial_utterance}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            return {"trigger": "wait", "reason": f"Fallback due to api error: {str(e)}"}
    
    try:
        decision = json.loads(response.choices[0].message.content)
        if decision.get("trigger") not in ["wait", "retrieve_now", "no_retrieval_needed"]:
            decision["trigger"] = "wait"
        return decision
    except Exception as e:
        return {"trigger": "wait", "reason": f"Fallback due to parse error: {str(e)}"}
