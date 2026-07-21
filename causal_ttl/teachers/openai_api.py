from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from .base import BaseTeacher
from .cache import TeacherCache


_RESPONSE_EXTRACTION_VERSION = "content_only_v1"


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
            "response_extraction_version": _RESPONSE_EXTRACTION_VERSION,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _extract_complete_json_object(text: str) -> Optional[str]:
        raw = (text or "").strip()
        if not raw:
            return None

        for start, char in enumerate(raw):
            if char != "{":
                continue
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(raw)):
                current = raw[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == '"':
                        in_string = False
                    continue

                if current == '"':
                    in_string = True
                    continue
                if current == "{":
                    depth += 1
                elif current == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start : index + 1].strip()
                        try:
                            json.loads(candidate)
                        except (json.JSONDecodeError, TypeError):
                            break
                        return candidate
        return None

    @staticmethod
    def _coerce_message_text(value: object) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, list):
            return ""

        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            content = item.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        return "".join(parts)

    @staticmethod
    def _preview_text(text: str, limit: int = 200) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

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
        content = self._coerce_message_text(message.get("content"))
        reasoning = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
        expects_json = bool(system_prompt and "Return only valid JSON" in system_prompt)

        if expects_json:
            content_json = self._extract_complete_json_object(content)
            if content_json:
                return content_json

            raise RuntimeError(
                "Teacher API JSON mode response did not contain a valid JSON object in message.content. "
                f"content_preview={self._preview_text(content)!r}; "
                f"reasoning_preview={self._preview_text(reasoning)!r}. "
                "reasoning_content is ignored for teacher outputs."
            )

        text = content.strip()
        if text:
            return text
        raise RuntimeError(
            "Teacher API returned an empty message.content. "
            f"reasoning_preview={self._preview_text(reasoning)!r}. "
            "reasoning_content is ignored for teacher outputs."
        )

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
