"""
Thin wrapper around the Anthropic Messages API.

Centralizing this means: one place to change models, one place to tune
retry/backoff, one place to enforce "must return valid JSON" for
structured extraction steps. Every other module calls `complete()` or
`complete_json()` and never touches the SDK directly.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import anthropic
import httpx

from app.config import Settings
from app.utils.logging_config import get_logger
from app.utils.resilience import retry_async

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.llm_provider == "anthropic":
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_seconds,
            )
        else:
            self._client = None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        format: Optional[str] = None,
    ) -> str:
        settings = self._settings

        async def _call_ollama() -> str:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                payload = {
                    "model": settings.local_llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature if temperature is not None else settings.llm_temperature
                    }
                }
                if format == "json":
                    payload["format"] = "json"

                response = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"].strip()

        if settings.llm_provider == "ollama":
            async def _call() -> str:
                async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                    payload = {
                        "model": settings.llm_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": temperature if temperature is not None else settings.llm_temperature
                        }
                    }
                    if format == "json":
                        payload["format"] = "json"

                    response = await client.post(
                        f"{settings.ollama_base_url}/api/chat",
                        json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["message"]["content"].strip()

            retryable = (httpx.HTTPError,)
        elif settings.llm_provider == "groq":
            async def _call() -> str:
                async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                    payload = {
                        "model": settings.llm_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "temperature": temperature if temperature is not None else settings.llm_temperature
                    }
                    if max_tokens or settings.llm_max_tokens:
                        payload["max_tokens"] = max_tokens or settings.llm_max_tokens
                    if format == "json":
                        payload["response_format"] = {"type": "json_object"}

                    headers = {
                        "Authorization": f"Bearer {settings.groq_api_key}",
                        "Content-Type": "application/json"
                    }
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

            retryable = (httpx.HTTPError,)
        else:
            async def _call() -> str:
                response = await self._client.messages.create(
                    model=settings.llm_model,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature if temperature is not None else settings.llm_temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text_parts = [block.text for block in response.content if block.type == "text"]
                return "".join(text_parts).strip()

            retryable = (
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.APIStatusError,
                anthropic.APITimeoutError,
            )

        def _on_retry(attempt: int, exc: BaseException) -> None:
            logger.warning("llm_retry attempt=%s provider=%s error=%s", attempt, settings.llm_provider, exc)

        try:
            return await retry_async(
                _call,
                max_attempts=settings.llm_max_retries,
                base_delay_seconds=1.5,
                retryable_exceptions=retryable,
                on_retry=_on_retry,
            )
        except Exception as exc:  # noqa: BLE001
            if settings.llm_provider == "groq":
                logger.error("groq_failed_falling_back_to_local error=%s", exc, exc_info=True)
                try:
                    def _on_local_retry(attempt: int, local_exc: BaseException) -> None:
                        logger.warning("local_llm_retry attempt=%s error=%s", attempt, local_exc)
                    return await retry_async(
                        _call_ollama,
                        max_attempts=settings.llm_max_retries,
                        base_delay_seconds=1.5,
                        retryable_exceptions=(httpx.HTTPError,),
                        on_retry=_on_local_retry,
                    )
                except Exception as local_exc:  # noqa: BLE001
                    raise LLMError(f"Both Groq and fallback local LLM failed. Local error: {local_exc}") from local_exc
            else:
                raise LLMError(f"LLM call failed after retries: {exc}") from exc

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Requests a JSON-only response and parses it defensively: models
        occasionally wrap JSON in markdown fences or add a stray
        preamble sentence despite instructions, so we strip fences and
        fall back to extracting the outermost {...} or [...] block.
        """
        raw = await self.complete(
            system=system + "\n\nRespond with ONLY valid JSON. No prose, no markdown fences.",
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            format="json",
        )
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if not match:
                raise LLMError(f"Model did not return parseable JSON: {raw[:300]}")
            return json.loads(match.group(1))
