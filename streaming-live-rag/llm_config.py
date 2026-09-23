"""
llm_config.py — Centralized LLM configuration and client handling (C1).

Single source of truth for the fast LLM provider (Groq).
- Fails loudly if model or credentials are missing/invalid.
- Retries ONLY transient errors (429, 500, 502, 503, timeouts) with backoff.
- Surfaces auth and configuration errors immediately.
- Defers client initialization to avoid crashes at import time.
"""

import os
import time
import random
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import threading

FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
_client = None
_client_lock = threading.Lock()


def get_groq_client():
    """Lazily initializes and returns the Groq client with thread safety."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                try:
                    from groq import Groq
                except ImportError:
                    raise ImportError("The 'groq' package is not installed. Please run: pip install groq")
                api_key = os.getenv("GROQ_API_KEY", "")
                if not api_key:
                    raise ValueError(
                        "GROQ_API_KEY environment variable is not set. "
                        "Provide a valid Groq API key in your .env file."
                    )
                _client = Groq(api_key=api_key)
    return _client


def assert_models_alive():
    """Verifies that the configured FAST_LLM_MODEL is currently served by Groq."""
    client = get_groq_client()
    ids = {x.id for x in client.models.list().data}
    if FAST_LLM_MODEL not in ids:
        raise RuntimeError(
            f"FAST_LLM_MODEL={FAST_LLM_MODEL!r} is not served by Groq. "
            f"Available models: {sorted(ids)[:8]}"
        )


def call_fast(**kwargs):
    """
    Executes a chat completion call with the configured FAST_LLM_MODEL.
    Retries only transient errors with exponential backoff and jitter.
    """
    client = get_groq_client()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=FAST_LLM_MODEL, **kwargs)
        except Exception as e:
            err_msg = str(e)
            is_transient = any(
                code in err_msg for code in ("429", "500", "502", "503", "timeout", "Rate limit")
            )
            if attempt == max_retries - 1 or not is_transient:
                raise
            time.sleep((2 ** attempt) + random.random())
