"""
One LLM adapter for every provider that speaks the OpenAI chat protocol
(Ollama, Gemini's OpenAI-compatible endpoint, OpenRouter, vLLM on RunPod).

make_llm_call("openrouter") returns a plain (prompt) -> str function, which is
exactly the shape of the `llm_call_fn` we inject everywhere else.
"""

import os
import threading
import time
from typing import Callable

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None,
               "model_env": "OLLAMA_MODEL", "default_model": "mistral:7b", "min_interval": 0.0},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
               "default_model": "gemini-3.8-flash", "min_interval": 13.0},
    # Free model ids change often: pick one ending in ":free" from
    # https://openrouter.ai/models?max_price=0 and export OPENROUTER_MODEL.
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY", "model_env": "OPENROUTER_MODEL",
                   "default_model": None, "min_interval": 4.0},
    "vllm":   {"base_url": None,  # read from VLLM_BASE_URL when the client is built
               "key_env": None, "model_env": "VLLM_MODEL", "default_model": None, "min_interval": 0.0},
}

MAX_RETRIES = 5


def make_llm_call(provider: str, model: str | None = None, temperature: float = 0.2,
                  max_tokens: int = 4096,
                  min_interval: float | None = None) -> Callable[[str], str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    base_url = cfg["base_url"] or os.environ["VLLM_BASE_URL"]
    api_key = os.environ[cfg["key_env"]] if cfg["key_env"] else "EMPTY"
    model = model or os.environ.get(cfg["model_env"], cfg["default_model"])
    if not model:
        raise ValueError(f"Set {cfg['model_env']} (or pass model=) for provider '{provider}'")

    # max_retries=0: we do our own backoff below so we can print what happens.
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=120)

    # Free tiers cap requests/minute: space calls out instead of hitting the cap.
    gap = cfg.get("min_interval", 0.0) if min_interval is None else min_interval
    if os.environ.get("LLM_MIN_INTERVAL"):   # override without editing code
        gap = float(os.environ["LLM_MIN_INTERVAL"])
    last_call = [0.0]
    lock = threading.Lock()  # FastAPI runs the pipeline in a thread pool: space calls safely

    def call(prompt: str) -> str:
        with lock:
            wait_for = gap - (time.monotonic() - last_call[0])
            if wait_for > 0:
                time.sleep(wait_for)
            last_call[0] = time.monotonic()
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                if not resp.choices:
                    # OpenRouter sometimes returns HTTP 200 with {"error": {...}} and no choices
                    # (upstream provider busy, rate limit, moderation). Treat it as transient.
                    err = str((getattr(resp, "model_extra", None) or {}).get("error", "no choices"))
                    if any(w in err.lower() for w in ("per-day", "per day", "daily")):
                        raise RuntimeError(f"{provider} daily quota exhausted")
                    wait = 2 ** attempt * 5
                    print(f"[{provider}] empty response ({err[:120]}), retry in {wait}s "
                          f"({attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                choice = resp.choices[0]
                text = choice.message.content or ""
                if not text:
                    # "length" = the model ran out of tokens (often while "thinking")
                    print(f"[{provider}] empty answer, finish_reason={choice.finish_reason}")
                return text
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                msg = str(e).lower()
                if isinstance(e, RateLimitError) and any(
                        w in msg for w in ("per day", "perday", "daily", "rpd")):
                    raise RuntimeError(f"{provider} daily quota exhausted") from e
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"[{provider}] {type(e).__name__}, retry in {wait}s "
                      f"({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
        raise RuntimeError(f"{provider} failed after {MAX_RETRIES} retries")

    return call


def make_llm_chain(specs: str, **kwargs) -> Callable[[str], str]:
    """
    Try providers in order; move on when one fails after its retries.
    specs is comma-separated, each "provider" or "provider=model":
        "gemini,openrouter=meta-llama/llama-3.3-70b-instruct:free"
    """
    calls = []
    for spec in specs.split(","):
        provider, _, model = spec.strip().partition("=")
        calls.append((provider, make_llm_call(provider, model=model or None, **kwargs)))

    def call(prompt: str) -> str:
        errors = []
        for provider, fn in calls:
            try:
                return fn(prompt)
            except RuntimeError as e:
                print(f"[chain] {provider} failed ({e}); trying next")
                errors.append(f"{provider}: {e}")
        raise RuntimeError("all providers failed: " + "; ".join(errors))

    return call


def describe_chain(specs: str) -> list[str]:
    """Human-readable "provider: model" for each entry, resolved the same way as make_llm_call."""
    out = []
    for spec in specs.split(","):
        provider, _, model = spec.strip().partition("=")
        cfg = PROVIDERS[provider]
        model = model or os.environ.get(cfg["model_env"], cfg["default_model"]) or "(model not set)"
        out.append(f"{provider}: {model}")
    return out