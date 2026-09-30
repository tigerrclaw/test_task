from __future__ import annotations

import time
from dataclasses import dataclass

from hotwork.models import facts_response_format


class LLMError(Exception):
    def __init__(self, message: str, raw: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.raw = raw
        self.retryable = retryable


@dataclass
class LLMCall:
    raw: str
    model: str
    latency_ms: float
    usage: dict | None


def is_retryable(exc: Exception) -> bool:
    """Сбой связи, таймаут одного запроса или временная недоступность провайдера."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    name = type(exc).__name__.lower()
    if any(token in name for token in ("timeout", "connection", "ratelimit", "connect")):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("timeout", "timed out", "connection", "connect error", "unavailable"))


def redact(text: str, secret: str | None) -> str:
    if not text or not secret:
        return text
    return text.replace(secret, "***")


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None

    def complete(self, system: str, user: str) -> LLMCall:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.perf_counter()
        try:
            response = self._create(messages, with_limits=True)
        except Exception as exc:
            lowered = redact(str(exc), self.api_key).lower()
            if "temperature" not in lowered and "max_tokens" not in lowered:
                raise LLMError(self._message(exc), retryable=is_retryable(exc)) from None
            try:
                response = self._create(messages, with_limits=False)
            except Exception as second:
                raise LLMError(self._message(second), retryable=is_retryable(second)) from None
        latency_ms = (time.perf_counter() - started) * 1000
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return LLMCall(
            raw=redact(content, self.api_key),
            model=self.model,
            latency_ms=latency_ms,
            usage=usage,
        )

    def _create(self, messages: list[dict[str, str]], with_limits: bool):
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "response_format": facts_response_format(),
        }
        if with_limits:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 800
        return self._client.chat.completions.create(**kwargs)

    def _message(self, exc: Exception) -> str:
        return f"{type(exc).__name__}: {redact(str(exc), self.api_key)}"
