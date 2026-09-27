"""
llm_config.py — Centralized Multi-Provider LLM Configuration & BYOK Client Manager.

Supports Bring Your Own Key (BYOK) for both Fast and Quality synthesis tiers:
  - Groq (Cloud ultra-low latency)
  - Ollama (100% Local / Free / Offline)
  - NVIDIA NIM (integrate.api.nvidia.com)
  - Google Gemini (Google AI Studio)
  - OpenAI & Custom OpenAI-compatible endpoints (vLLM, LM Studio, Together, DeepSeek)

Thread-safe dynamic configuration without server restart.
"""

import os
import time
import random
import threading
import logging
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider Presets & Default Configurations
# ---------------------------------------------------------------------------

PROVIDER_PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_fast_model": "llama-3.1-8b-instant",
        "default_synthesis_model": "llama-3.3-70b-versatile",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_fast_model": "llama3.2",
        "default_synthesis_model": "llama3.1",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_fast_model": "meta/llama-3.1-8b-instruct",
        "default_synthesis_model": "meta/llama-3.3-70b-instruct",
    },
    "gemini": {
        "base_url": "",
        "default_fast_model": "gemini-3.5-flash-lite",
        "default_synthesis_model": "gemini-3.5-flash-lite",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_fast_model": "gpt-4o-mini",
        "default_synthesis_model": "gpt-4o",
    },
    "custom": {
        "base_url": "http://localhost:8000/v1",
        "default_fast_model": "default-model",
        "default_synthesis_model": "default-model",
    },
}

_config_lock = threading.RLock()

_fast_prov = os.getenv("FAST_LLM_PROVIDER", "groq").strip().lower()
if _fast_prov not in PROVIDER_PRESETS:
    logger.warning("Unsupported FAST_LLM_PROVIDER '%s'; falling back to 'groq'", _fast_prov)
    _fast_prov = "groq"
elif _fast_prov == "gemini":
    logger.warning("Gemini cannot be FAST_LLM_PROVIDER; falling back to 'groq'")
    _fast_prov = "groq"

_synth_prov = os.getenv("SYNTHESIS_LLM_PROVIDER", "gemini").strip().lower()
if _synth_prov not in PROVIDER_PRESETS:
    logger.warning("Unsupported SYNTHESIS_LLM_PROVIDER '%s'; falling back to 'gemini'", _synth_prov)
    _synth_prov = "gemini"

_current_config: Dict[str, Any] = {
    "fast_provider": _fast_prov,
    "fast_api_key": os.getenv("GROQ_API_KEY", ""),
    "fast_base_url": os.getenv("FAST_LLM_BASE_URL", PROVIDER_PRESETS[_fast_prov]["base_url"]),
    "fast_model": os.getenv("FAST_LLM_MODEL", PROVIDER_PRESETS[_fast_prov]["default_fast_model"]),
    
    "synthesis_provider": _synth_prov,
    "synthesis_api_key": os.getenv("GEMINI_API_KEY", ""),
    "synthesis_base_url": os.getenv("SYNTHESIS_LLM_BASE_URL", PROVIDER_PRESETS[_synth_prov]["base_url"]),
    "synthesis_model": os.getenv("SYNTHESIS_LLM_MODEL", PROVIDER_PRESETS[_synth_prov]["default_synthesis_model"]),
}

# Cached client instances
_fast_client = None
_synthesis_client = None
_gemini_client = None


def redact_keys(text: str) -> str:
    """Scrubs any configured API keys from text or error messages to prevent credential leakage."""
    if not text:
        return ""
    scrubbed = str(text)
    with _config_lock:
        for k in (_current_config.get("fast_api_key", ""), _current_config.get("synthesis_api_key", "")):
            if k and len(k) >= 4:
                scrubbed = scrubbed.replace(k, "[REDACTED]")
    return scrubbed


def get_llm_config(mask_secrets: bool = True) -> Dict[str, Any]:
    """
    Returns current LLM configuration adhering to write-only keys architecture:
    zero key material is ever returned. Only boolean presence is disclosed.
    """
    with _config_lock:
        return {
            "fast_provider": _current_config["fast_provider"],
            "fast_model": _current_config["fast_model"],
            "fast_base_url": _current_config["fast_base_url"],
            "fast_key_configured": bool(_current_config.get("fast_api_key")),
            "synthesis_provider": _current_config["synthesis_provider"],
            "synthesis_model": _current_config["synthesis_model"],
            "synthesis_base_url": _current_config["synthesis_base_url"],
            "synthesis_key_configured": bool(_current_config.get("synthesis_api_key")),
        }


def clear_llm_key(slot: str) -> bool:
    """Explicitly removes a configured key (write-only security model)."""
    global _fast_client, _synthesis_client, _gemini_client
    with _config_lock:
        if slot == "fast":
            _current_config["fast_api_key"] = ""
            _fast_client = None
            return True
        elif slot == "synthesis":
            _current_config["synthesis_api_key"] = ""
            _synthesis_client = None
            _gemini_client = None
            return True
        return False


def update_llm_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Updates runtime configuration with provider normalization and model reconciliation."""
    global _fast_client, _synthesis_client, _gemini_client
    with _config_lock:
        # Check fast provider restrictions
        if "fast_provider" in updates and updates["fast_provider"] is not None:
            new_fp = str(updates["fast_provider"]).strip().lower()
            if new_fp not in PROVIDER_PRESETS:
                raise ValueError(f"Unsupported fast provider '{new_fp}'. Supported providers: {sorted(PROVIDER_PRESETS.keys())}")
            if new_fp == "gemini":
                raise ValueError("Gemini is not supported as the fast streaming controller tier; use Groq, Ollama, NVIDIA, or OpenAI.")
            old_fp = _current_config["fast_provider"]
            _current_config["fast_provider"] = new_fp
            # Reconcile model and base_url if provider changed and no explicit model/base_url provided
            if new_fp != old_fp:
                if "fast_model" not in updates:
                    _current_config["fast_model"] = PROVIDER_PRESETS[new_fp].get("default_fast_model", _current_config["fast_model"])
                if "fast_base_url" not in updates:
                    _current_config["fast_base_url"] = PROVIDER_PRESETS[new_fp].get("base_url", _current_config["fast_base_url"])

        if "synthesis_provider" in updates and updates["synthesis_provider"] is not None:
            new_sp = str(updates["synthesis_provider"]).strip().lower()
            if new_sp not in PROVIDER_PRESETS:
                raise ValueError(f"Unsupported synthesis provider '{new_sp}'. Supported providers: {sorted(PROVIDER_PRESETS.keys())}")
            old_sp = _current_config["synthesis_provider"]
            _current_config["synthesis_provider"] = new_sp
            if new_sp != old_sp:
                if "synthesis_model" not in updates:
                    _current_config["synthesis_model"] = PROVIDER_PRESETS[new_sp].get("default_synthesis_model", _current_config["synthesis_model"])
                if "synthesis_base_url" not in updates:
                    _current_config["synthesis_base_url"] = PROVIDER_PRESETS[new_sp].get("base_url", _current_config["synthesis_base_url"])

        for k in (
            "fast_api_key", "fast_base_url", "fast_model",
            "synthesis_api_key", "synthesis_base_url", "synthesis_model"
        ):
            if k in updates and updates[k] is not None:
                val = str(updates[k]).strip()
                if k.endswith("_api_key") and not val:
                    continue  # empty string does not overwrite key; use clear_llm_key to remove
                _current_config[k] = val

        # Invalidate cached clients so new credentials apply immediately
        _fast_client = None
        _synthesis_client = None
        _gemini_client = None

        return get_llm_config()


# ---------------------------------------------------------------------------
# Client Factories
# ---------------------------------------------------------------------------

def _build_openai_compatible_client(provider: str, api_key: str, base_url: str):
    """Builds an OpenAI-compatible client for Groq, Ollama, NVIDIA, OpenAI, or custom hosts."""
    from openai import OpenAI
    key = api_key or "local-key"
    url = (base_url or "").strip()
    if not url:
        preset = PROVIDER_PRESETS.get(provider, {})
        url = preset.get("base_url") or "https://api.openai.com/v1"

    return OpenAI(base_url=url, api_key=key, timeout=30.0)


def get_fast_client():
    """Returns client for Fast Tier (Streaming Controller & Decomposer)."""
    global _fast_client
    if _fast_client is None:
        with _config_lock:
            if _fast_client is None:
                p = _current_config["fast_provider"]
                k = _current_config["fast_api_key"]
                url = _current_config["fast_base_url"]
                
                # Check for legacy Groq client if requested
                if p == "groq" and not url:
                    try:
                        from groq import Groq
                        _fast_client = Groq(api_key=k)
                        return _fast_client
                    except ImportError:
                        pass
                
                _fast_client = _build_openai_compatible_client(p, k, url)
    return _fast_client


def get_synthesis_client():
    """Returns client for Quality Tier (Final Grounded Synthesis)."""
    global _synthesis_client, _gemini_client
    p = _current_config["synthesis_provider"]
    k = _current_config["synthesis_api_key"]
    url = _current_config["synthesis_base_url"]

    if p == "gemini":
        if _gemini_client is None:
            with _config_lock:
                if _gemini_client is None:
                    try:
                        from google import genai
                        _gemini_client = genai.Client(api_key=k, http_options={"timeout": 60_000})
                    except (ImportError, Exception):
                        # Fallback to OpenAI-compatible Gemini endpoint if google-genai package missing/fails
                        from openai import OpenAI
                        _gemini_client = OpenAI(
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                            api_key=k,
                            timeout=60.0
                        )
        return _gemini_client

    if _synthesis_client is None:
        with _config_lock:
            if _synthesis_client is None:
                _synthesis_client = _build_openai_compatible_client(p, k, url)
    return _synthesis_client


# ---------------------------------------------------------------------------
# Invocation Wrappers
# ---------------------------------------------------------------------------

# Backward-compat alias for existing imports
FAST_LLM_MODEL = _current_config["fast_model"]


def assert_models_alive():
    """Verifies that the configured Fast LLM model is reachable."""
    res = test_llm_connection(target="fast")
    if not res.get("fast", {}).get("ok"):
        raise RuntimeError(f"Fast LLM model check failed: {res.get('fast', {}).get('error')}")


def call_fast(**kwargs):
    """
    Executes a chat completion call with the configured Fast LLM.
    Retries transient errors with exponential backoff.
    Reads client and model atomically under _config_lock to avoid torn-read races.
    """
    with _config_lock:
        client = get_fast_client()
        model = _current_config["fast_model"]
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except Exception as e:
            err_msg = str(e)
            is_transient = any(
                code in err_msg for code in ("429", "500", "502", "503", "timeout", "Rate limit", "ResourceExhausted")
            )
            if attempt == max_retries - 1 or not is_transient:
                raise
            time.sleep((2 ** attempt) + random.random())


def call_synthesis(prompt: str) -> Dict[str, Any]:
    """
    Unified synthesis caller routing to Gemini, NVIDIA, Ollama, Groq, or OpenAI.
    Reads provider, model, and client atomically under _config_lock.
    Returns:
      {
        "text": str,
        "input_tokens": int,
        "output_tokens": int
      }
    """
    with _config_lock:
        provider = _current_config["synthesis_provider"]
        model = _current_config["synthesis_model"]
        client = get_synthesis_client()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                # Check if native google-genai or OpenAI wrapper
                if hasattr(client, "models"):
                    from google.genai import types
                    try:
                        res = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=0.0)
                        )
                    except Exception as ge:
                        ge_str = str(ge)
                        if ("429" in ge_str or "RESOURCE_EXHAUSTED" in ge_str) and model != "gemini-3.5-flash-lite":
                            logger.warning("Gemini model %s hit quota/429; failing over to gemini-3.5-flash-lite", model)
                            res = client.models.generate_content(
                                model="gemini-3.5-flash-lite",
                                contents=prompt,
                                config=types.GenerateContentConfig(temperature=0.0)
                            )
                        elif ("429" in ge_str or "RESOURCE_EXHAUSTED" in ge_str) and model == "gemini-3.5-flash-lite":
                            logger.warning("Gemini model %s hit quota/429; failing over to gemini-3.6-flash", model)
                            res = client.models.generate_content(
                                model="gemini-3.6-flash",
                                contents=prompt,
                                config=types.GenerateContentConfig(temperature=0.0)
                            )
                        else:
                            raise
                    txt = (res.text or "").strip()
                    usage = getattr(res, "usage_metadata", None)
                    in_tok = getattr(usage, "prompt_token_count", 0) or 0
                    out_tok = getattr(usage, "candidates_token_count", 0) or 0
                else:
                    # OpenAI-compatible Gemini endpoint
                    res = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    txt = res.choices[0].message.content or ""
                    in_tok = getattr(res.usage, "prompt_tokens", 0) or 0
                    out_tok = getattr(res.usage, "completion_tokens", 0) or 0

                return {"text": txt, "input_tokens": in_tok, "output_tokens": out_tok}

            else:
                # Any OpenAI-compatible provider (NVIDIA NIM, Ollama, Groq, OpenAI)
                res = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                txt = (res.choices[0].message.content or "").strip()
                in_tok = getattr(res.usage, "prompt_tokens", 0) or 0
                out_tok = getattr(res.usage, "completion_tokens", 0) or 0
                return {"text": txt, "input_tokens": in_tok, "output_tokens": out_tok}

        except Exception as e:
            err_msg = str(e)
            is_transient = any(
                c in err_msg for c in ("429", "500", "502", "503", "504", "ResourceExhausted", "Timeout", "timeout")
            )
            if attempt == max_retries - 1 or not is_transient:
                raise
            time.sleep((2 ** attempt) + random.random())


def test_llm_connection(target: str = "both") -> Dict[str, Any]:
    """Pings configured providers with a minimal prompt to verify credentials and measure latency."""
    report = {}

    if target in ("fast", "both"):
        t0 = time.time()
        try:
            with _config_lock:
                client = get_fast_client()
                m = _current_config["fast_model"]
                p = _current_config["fast_provider"]
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Ping. Respond with 'OK'."}],
                max_tokens=5,
            )
            ms = round((time.time() - t0) * 1000, 1)
            report["fast"] = {
                "ok": True,
                "provider": p,
                "model": m,
                "latency_ms": ms,
                "response": (res.choices[0].message.content or "").strip(),
            }
        except Exception as e:
            report["fast"] = {
                "ok": False,
                "provider": _current_config["fast_provider"],
                "model": _current_config["fast_model"],
                "error": redact_keys(str(e)[:300]),
            }

    if target in ("synthesis", "both"):
        t0 = time.time()
        try:
            res = call_synthesis("Ping. Respond with 'OK'.")
            ms = round((time.time() - t0) * 1000, 1)
            with _config_lock:
                p = _current_config["synthesis_provider"]
                m = _current_config["synthesis_model"]
            report["synthesis"] = {
                "ok": True,
                "provider": p,
                "model": m,
                "latency_ms": ms,
                "response": res["text"][:30],
            }
        except Exception as e:
            report["synthesis"] = {
                "ok": False,
                "provider": _current_config["synthesis_provider"],
                "model": _current_config["synthesis_model"],
                "error": redact_keys(str(e)[:300]),
            }

    return report
