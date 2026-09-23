import os
import time
from dotenv import load_dotenv

try:
    from groq import Groq
    from google import genai
except ImportError:
    print("Please install the dependencies: pip install groq google-genai")
    exit(1)

load_dotenv()

def run_groq_test(prompt: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("[Groq Error] GROQ_API_KEY missing in .env")
        return
        
    client = Groq(api_key=api_key)
    model = os.getenv("FAST_LLM_MODEL", "groq/compound-mini")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        total_time = (time.time() - start_time) * 1000
        print("[Groq - Fast Controller Task]")
        print(f"Response: {response.choices[0].message.content.strip()}")
        print(f"Latency: {total_time:.2f} ms\n")
    except Exception as e:
        print(f"[Groq Error]: {e}\n")

def run_gemini_test(prompt: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Gemini Error] GEMINI_API_KEY missing in .env")
        return
        
    client = genai.Client(api_key=api_key)
    model = os.getenv("SYNTHESIS_LLM_MODEL", "gemini-3.8-flash")
    
    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        total_time = (time.time() - start_time) * 1000
        print("[Gemini - Synthesis Task]")
        print(f"Response: {response.text.strip()}")
        print(f"Latency: {total_time:.2f} ms\n")
    except Exception as e:
        print(f"[Gemini Error]: {e}\n")

def main():
    print("--- Running Dual-Provider LLM POC ---\n")
    
    fast_prompt_1 = "Should we retrieve external documents for: 'Hello there'? Reply strictly with YES or NO."
    fast_prompt_2 = "Generate a JSON response indicating { 'retrieve': true }."
    
    synthesis_prompt = "Synthesize an answer using this document: [Doc_01 §1] Pune capacity is 30. User Query: What is the capacity?"
    
    run_groq_test(fast_prompt_1)
    run_groq_test(fast_prompt_2)
    run_gemini_test(synthesis_prompt)

if __name__ == "__main__":
    main()
