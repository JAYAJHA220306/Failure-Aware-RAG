"""LLM client for any OpenAI-compatible chat endpoint, plus a no-key 'mock' mode.

Works with OpenAI, Groq, Together, OpenRouter, and local servers such as Ollama
(LLM_BASE_URL=http://localhost:11434/v1) or LM Studio.
"""
from __future__ import annotations

import time

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.calls = 0
        self.tokens = 0

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def reset_counters(self) -> None:
        self.calls = 0
        self.tokens = 0

    def chat(self, system: str, user: str, max_tokens: int = 600) -> str:
        if self.is_mock:
            raise LLMError("chat() is not available in mock mode; callers use extractive fallbacks.")
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        body = {
            "model": self.model,
            "temperature": settings.llm_temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_s)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                self.calls += 1
                self.tokens += int(data.get("usage", {}).get("total_tokens", 0))
                return data["choices"][0]["message"]["content"].strip()
            except (httpx.HTTPError, LLMError, KeyError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM request failed: {last_error}")

    def describe(self) -> str:
        return "Mock (extractive, no API key)" if self.is_mock else f"{self.model} via {settings.llm_base_url}"
