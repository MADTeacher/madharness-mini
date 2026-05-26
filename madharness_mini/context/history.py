"""Атомарные элементы истории диалога для слоя контекста."""

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HistoryEntry:
    """Один элемент истории: обычный ответ или assistant+tool results."""

    kind: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_call_ids: set[str] = field(default_factory=set)
    seen_tool_call_ids: set[str] = field(default_factory=set)
    pending_followups: list[dict[str, Any]] = field(default_factory=list)

    def rendered_messages(self) -> list[dict[str, Any]]:
        """Отдаём tool follow-ups только после закрытия всех tool_calls."""

        rendered = [copy.deepcopy(message) for message in self.messages]
        if self.expected_tool_call_ids <= self.seen_tool_call_ids:
            rendered.extend(copy.deepcopy(self.pending_followups))
        return rendered
