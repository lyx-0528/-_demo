from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from .base import BaseTeacher
from .cache import TeacherCache


class OpenAICompatibleTeacher(BaseTeacher):
    def __init__(
        self,
        api_base: str,
        api_key: Optional[str],
        model: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = TeacherCache(cache_dir)

    def _cache_key(self, prompt: str, system_prompt: Optional[str]) -> str:
        payload = {
            "api_base": self.api_base,
            "model": self.model,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "system_prompt": system_prompt,
            "prompt": prompt,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _request(self, prompt: str, system_prompt: Optional[str]) -> str:
        url = f"{self.api_base}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        if system_prompt and "Return only valid JSON" in system_prompt:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        choices = response_payload.get("choices") or []
        if not choices:
            raise RuntimeError("Teacher API returned no choices.")

        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        reasoning = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
        text = content.strip()
        if not text and reasoning.strip():
            text = reasoning.strip()
        elif reasoning.strip() and reasoning.strip() not in text:
            text = f"{text}\n{reasoning.strip()}".strip()
        if not text:
            raise RuntimeError("Teacher API returned an empty response.")
        return text

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        key = self._cache_key(prompt, system_prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._request(prompt, system_prompt)
                self.cache.set(
                    key,
                    prompt,
                    response,
                    {
                        "api_base": self.api_base,
                        "model": self.model,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_new_tokens,
                    },
                )
                return response
            except (urllib.error.URLError, TimeoutError, RuntimeError) as err:
                last_error = err
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Teacher API call failed after {self.max_retries} retries: {last_error}")
