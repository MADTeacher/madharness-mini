"""Атомарные элементы истории диалога для слоя контекста."""

import copy
from dataclasses import dataclass, field
from typing import Any

# Заглушка вместо base64-картинки, которая уже ушла в API в прошлом запросе:
# держать вложение в истории на каждом ходу дорого, а модели оно нужно один раз.
IMAGE_OMITTED_PLACEHOLDER = "[image omitted from history after first send]"


@dataclass
class HistoryEntry:
    """Один элемент истории: обычный ответ или assistant+tool results."""

    kind: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_call_ids: set[str] = field(default_factory=set)
    seen_tool_call_ids: set[str] = field(default_factory=set)
    pending_followups: list[dict[str, Any]] = field(default_factory=list)
    # True, как только запрос с этим элементом реально ушёл модели: тяжёлые
    # вложения (base64) уже получены, дальше держим лишь текстовую пометку.
    sent: bool = False

    def rendered_messages(self) -> list[dict[str, Any]]:
        """Отдаём tool follow-ups только после закрытия всех tool_calls."""

        rendered = [copy.deepcopy(message) for message in self.messages]
        if self.expected_tool_call_ids <= self.seen_tool_call_ids:
            rendered.extend(copy.deepcopy(self.pending_followups))
        if self.sent:
            for message in rendered:
                _strip_image_parts(message)
        return rendered


def _strip_image_parts(message: dict[str, Any]) -> None:
    """Заменяет уже отправленные image_url-части на текстовую заглушку.

    Картинка нужна модели только в первом запросе после read_image; дальше
    держать её в истории дорого. Замена применяется лишь к уже отправленным
    элементам, поэтому base64 реально уходит в API ровно один раз.
    """

    content = message.get("content")
    if not isinstance(content, list):
        return
    message["content"] = [
        {"type": "text", "text": IMAGE_OMITTED_PLACEHOLDER}
        if isinstance(part, dict) and part.get("type") == "image_url"
        else part
        for part in content
    ]
