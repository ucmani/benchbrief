import os
import requests

VLLM_BASE_URL_ENV = "VLM_BASE_URL"
DEFAULT_TIMEOUT_SECONDS = 60

def call_vllm(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    base_url = os.environ.get(VLLM_BASE_URL_ENV)
    if not base_url:
        raise RuntimeError(f"{VLLM_BASE_URL_ENV} environment variable not set")

    response = requests.post(
        f"{base_url}/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["text"]