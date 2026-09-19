import os
import time
from dotenv import load_dotenv

# Run `pip install google-genai` before executing this script
try:
    from google import genai
except ImportError:
    print("Please install the Gemini SDK: pip install google-genai")
    exit(1)

load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Please set GEMINI_API_KEY in .env")
        return
        
    client = genai.Client(api_key=api_key)
    model = os.getenv("FAST_LLM_MODEL", "gemini-3.8-flash")
    
    prompts = [
        "What is the venue capacity for Pune?",
        "Extract intent from: 'I need to cancel my flight to'",
        "Should we retrieve for: 'Hello there'",
        "Decompose into sub-queries: 'What is the refund policy and who approves travel?'",
        "Generate a JSON response indicating { 'retrieve': true }."
    ]
    
    print(f"Running LLM POC against Gemini using model: {model}")
    for i, p in enumerate(prompts):
        print(f"\n--- Prompt {i+1} ---")
        start_time = time.time()
        
        try:
            response = client.models.generate_content(
                model=model,
                contents=p,
            )
            
            total_time = (time.time() - start_time) * 1000
            content = response.text
            
            print(f"Response: {content.strip()}")
            print(f"Total Latency: {total_time:.2f} ms")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
