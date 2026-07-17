from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional


class TeacherCache:
    def __init__(self, cache_dir: Optional[str]) -> None:
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if path is None or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError, ValueError):
            return None
        return payload.get("response")

    def set(self, key: str, prompt: str, response: str, metadata: dict[str, Any]) -> None:
        path = self._path(key)
        if path is None:
            return
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"prompt": prompt, "response": response, "metadata": metadata},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp_path, path)
