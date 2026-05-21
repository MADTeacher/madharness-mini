"""HTTP-клиент к OpenAI-совместимому Chat Completions API."""

import datetime as dt
import json
import math
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any

from .config import Config


class ModelRateLimitError(RuntimeError):
    """Провайдер вернул HTTP 429; в атрибутах — Retry-After для повтора."""

    def __init__(
        self,
        *,
        status: int,
        body: str,
        retry_after: str | None,
        retry_after_seconds: int | None,
    ):
        self.status = status
        self.status_code = status
        self.body = body
        self.retry_after = retry_after
        self.retry_after_seconds = retry_after_seconds
        message = (
            "достигнут лимит LLM API (HTTP 429); попробуйте позже, "
            "смените модель/ключ или проверьте лимиты провайдера"
        )
        super().__init__(message)


def parse_retry_after(value: str | None) -> int | None:
    """Переводим заголовок Retry-After в секунды ожидания.

    Поддерживаем число секунд или HTTP-date; неразборное — None.
    """

    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.isdecimal():
        return int(raw)
    try:
        retry_at = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    return max(0, math.ceil((retry_at - now).total_seconds()))


class ModelClient:
    """Отправляет POST /chat/completions и возвращает сырой JSON ответа API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def settings(self) -> dict[str, Any]:
        """Достаём из Config URL, ключ, модель и доп. HTTP-заголовки.

        Нужен отдельным методом, чтобы chat() занимался только запросом.
        """

        headers = self.cfg.data.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        data = {
            "base_url": self.cfg.data.get("base_url") or "https://openrouter.ai/api/v1",
            "api_key": self.cfg.data.get("api_key") or "",
            "headers": dict(headers),
            "model": self.cfg.data.get("model"),
        }
        return data

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Один вызов /chat/completions: сообщения и опционально схемы tools.

        Без api_key — RuntimeError с подсказкой про init/env.
        При HTTP-ошибке тело ответа попадает в текст исключения для CLI и трассы.
        429 выделяем в ModelRateLimitError для повтора в loop.py.
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
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                raise ModelRateLimitError(
                    status=exc.code,
                    body=body,
                    retry_after=retry_after,
                    retry_after_seconds=parse_retry_after(retry_after),
                ) from exc
            raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
