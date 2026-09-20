import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "groq/compound-mini")

def decide_retrieval(partial_utterance: str) -> dict:
    """
    Classifies a partial utterance into one of three states:
    1. "wait" - The user hasn't finished expressing the core entity or question.
    2. "retrieve_now" - The user has provided enough context to search the database.
    3. "no_retrieval_needed" - The user is just chatting or asked a question that doesn't need external data.
    """
    system_prompt = """
    You are a real-time speech controller evaluating a live stream of text as a user speaks.
    Your job is to decide if we have enough information to execute a search against our database yet.
    
    Output strictly in JSON format:
    {
        "trigger": "wait" | "retrieve_now" | "no_retrieval_needed",
        "reason": "short explanation"
    }
    
    Rules for Triggering:
    1. retrieve_now (AGGRESSIVE EARLY RETRIEVAL): 
       Trigger AS SOON AS a strong entity, noun phrase, or clear search intent is visible, EVEN IF the sentence is grammatically incomplete. 
       - "What is the maximum capacity of" -> retrieve_now (keyword "maximum capacity")
       - "I need to travel to Pune for the" -> retrieve_now ("travel to Pune")
       - "Who needs to approve international" -> retrieve_now ("approve international")
       - "What is the hotel" -> retrieve_now (keyword "hotel")
       - "Who handles the projector" -> retrieve_now (keyword "projector")
    
    2. no_retrieval_needed (CONVERSATIONAL / PRESENTATION):
       Trigger for greetings, pleasantries, generic requests for help, or meeting management chatter where no factual lookup is needed.
       - "Hello, how are you", "Can you help me?", "Let me pull up my slides", "Can everyone see my screen ok", "Thanks for joining the call today", "Moving on to the next topic", "I'll take questions at the end." -> no_retrieval_needed.
       
    3. wait:
       Trigger ONLY IF the user has barely started speaking and no distinct subject or intent has emerged yet.
       - "I was wondering about the..." -> wait.
       - "Give me the" -> wait.
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
