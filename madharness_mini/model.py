"""Минимальный клиент OpenAI-совместимого Chat Completions API."""

import json
import urllib.error
import urllib.request
from typing import Any

from .config import Config


class ModelClient:
    """Отправляет сообщения модели и возвращает сырой JSON-ответ API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def settings(self) -> dict[str, Any]:
        """Собрать настройки подключения из `Config`.

        Метод нормализует необязательные HTTP-заголовки и подставляет базовый
        URL по умолчанию, чтобы `chat` оставался узким транспортным методом.
        """

        headers = self.cfg.data.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        data = {
            "base_url": self.cfg.data.get("base_url")
            or "https://openrouter.ai/api/v1",
            "api_key": self.cfg.data.get("api_key") or "",
            "headers": dict(headers),
            "model": self.cfg.data.get("model"),
        }
        return data

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Вызвать `/chat/completions` с сообщениями и необязательными tools.

        Если ключ API не задан, метод поднимает `RuntimeError` с подсказкой для
        пользователя. При HTTP-ошибке тело ответа сохраняется в тексте ошибки,
        чтобы проблему было видно в CLI и трассе.
        """

        settings = self.settings()
        key = settings.get("api_key", "")
        if not key:
            raise RuntimeError(
                "нет ключа API для LLM API: запустите "
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
            raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
