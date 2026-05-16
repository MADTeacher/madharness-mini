import json
import urllib.error
import urllib.request
from typing import Any

from .config import Config


class ModelClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def provider(self) -> tuple[str, dict[str, Any]]:
        name = self.cfg.data.get("provider", "openrouter")
        providers = self.cfg.data.get("providers") or {}
        data = dict(providers.get(name) or {})
        if self.cfg.data.get("base_url"):
            data["base_url"] = self.cfg.data["base_url"]
        data.setdefault("base_url", "https://openrouter.ai/api/v1")
        if self.cfg.data.get("api_key"):
            data["api_key"] = self.cfg.data["api_key"]
        data.setdefault("api_key", "")
        data.setdefault("headers", {})
        return name, data

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        provider, settings = self.provider()
        key = settings.get("api_key", "")
        if not key:
            raise RuntimeError(
                f"нет ключа API для {provider}: запустите "
                "madharness-mini init --api-key ... или задайте MADHARNESS_MINI_API_KEY"
            )
        payload: dict[str, Any] = {
            "model": settings.get("model") or self.cfg.data["model"],
            "messages": messages,
            "temperature": self.cfg.data["temperature"],
        }
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
        headers = {"Content-Type": "application/json", **settings.get("headers", {})}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            settings["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{provider} HTTP {exc.code}: {body}") from exc
