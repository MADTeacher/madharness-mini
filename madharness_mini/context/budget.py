"""Оценка и обрезка контекста по длине JSON-сообщений."""

import json
from typing import Any

from .history import HistoryEntry


def messages_chars(messages: list[dict[str, Any]]) -> int:
    """Оцениваем размер контекста по компактному JSON-представлению."""

    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def clip_tool_messages(entries: list[HistoryEntry], limit: int) -> bool:
    """Укорачиваем content у role=tool сообщений перед отправкой модели."""

    changed = False
    for entry in entries:
        for message in entry.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or len(content) <= limit:
                continue
            message["content"] = clip_tool_content(content, limit)
            changed = True
    return changed


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
