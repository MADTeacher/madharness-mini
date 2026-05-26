"""Рендерим фрагменты и историю в сообщения Chat Completions."""

from typing import Any

from .fragments import ContextFragment
from .history import HistoryEntry


def render_messages(
    user_task: str,
    fragments: list[ContextFragment],
    entries: list[HistoryEntry],
) -> list[dict[str, Any]]:
    """Склеиваем system/user/history сообщения для запроса к модели."""

    messages: list[dict[str, Any]] = []
    system_parts = [
        fragment.text.rstrip()
        for fragment in fragments
        if fragment.placement == "system" and fragment.text.strip()
    ]
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    for fragment in fragments:
        if fragment.placement != "user" or not fragment.text.strip():
            continue
        messages.append({"role": "user", "content": fragment.text.rstrip()})
    messages.append({"role": "user", "content": user_task})
    for entry in entries:
        messages.extend(entry.rendered_messages())
    return messages
