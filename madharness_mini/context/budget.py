"""Оценка токенового бюджета и обрезка больших tool-наблюдений."""

import json
from typing import Any

from .history import HistoryEntry

# Консервативная цена токена без tokenizer конкретной модели.
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 3


def estimate_tokens(payload: Any) -> int:
    """Оцениваем токены по размеру компактного JSON в UTF-8.

    Харнесс работает с OpenAI-совместимыми провайдерами, где tokenizer зависит
    от выбранной модели. Поэтому бюджет намеренно использует один простой
    приближённый счётчик без runtime-зависимостей.
    """

    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return (
        len(raw) + TOKEN_ESTIMATE_BYTES_PER_TOKEN - 1
    ) // TOKEN_ESTIMATE_BYTES_PER_TOKEN


def estimate_request_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Оцениваем части запроса, которые занимают контекст модели."""

    payload: dict[str, Any] = {"messages": messages}
    messages_tokens = estimate_tokens(messages)
    tools_tokens = 0
    if tools:
        payload["tools"] = tools
        tools_tokens = estimate_tokens(tools)
    return {
        "messages_tokens_estimate": messages_tokens,
        "tools_tokens_estimate": tools_tokens,
        "request_tokens_estimate": estimate_tokens(payload),
    }


def clip_tool_messages(entries: list[HistoryEntry], limit: int) -> list[dict[str, Any]]:
    """Укорачиваем content у role=tool сообщений и описываем обрезанные места."""

    clipped: list[dict[str, Any]] = []
    for entry in entries:
        for message in entry.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or len(content) <= limit:
                continue
            shortened = clip_tool_content(content, limit)
            message["content"] = shortened
            clipped.append(
                {
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "before_chars": len(content),
                    "after_chars": len(shortened),
                    "saved_chars": len(content) - len(shortened),
                }
            )
    return clipped


def clip_tool_content(content: str, limit: int) -> str:
    """Сохраняем краткое JSON-наблюдение, когда полный output слишком велик."""

    excerpt = clip_text(content, max(40, limit // 2))
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return clip_text(content, limit)
    if not isinstance(payload, dict):
        return clip_text(content, limit)
    compact = {key: payload[key] for key in ("ok", "tool", "summary") if key in payload}
    compact["_context_truncated"] = True
    compact["content_excerpt"] = excerpt
    rendered = json.dumps(compact, ensure_ascii=False)
    if len(rendered) <= limit:
        return rendered
    return clip_text(content, limit)


def clip_text(text: str, limit: int) -> str:
    """Обрезаем строку с явной пометкой для модели."""

    if len(text) <= limit:
        return text
    marker = f"\n...[context clipped {len(text) - limit} chars]"
    keep = max(limit - len(marker), 0)
    return text[:keep] + marker
