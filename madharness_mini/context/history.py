"""Атомарные элементы истории диалога для слоя контекста."""

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileRef:
    """Описание файлового эффекта одного вызова инструмента.

    Слой контекста получает от model_loop для read_file/write_file/apply_patch
    ссылки на затронутые файлы: путь, тип воздействия и при возможности хэш
    содержимого. По ним строится файловый реестр, который кормит напоминание о
    «грязных» файлах (гипотеза C) и дедуп tool_output (гипотеза D).
    """

    path: str
    kind: str  # "read" | "write" | "patch"
    content_hash: str | None = None


@dataclass
class HistoryEntry:
    """Один элемент истории: обычный ответ или assistant+tool results."""

    kind: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_call_ids: set[str] = field(default_factory=set)
    seen_tool_call_ids: set[str] = field(default_factory=set)
    pending_followups: list[dict[str, Any]] = field(default_factory=list)
    # Файловые эффекты этого хода: нужны для дедупа и напоминания о состоянии.
    file_refs: list[FileRef] = field(default_factory=list)

    def rendered_messages(self) -> list[dict[str, Any]]:
        """Отдаём tool follow-ups только после закрытия всех tool_calls.

        Служебные поля вида '_malformed_arguments' (маркер битого JSON из
        _sanitize_tool_calls) нужны только для диагностики внутри harness — в
        реальный запрос к модели они попасть не должны, провайдер не ждёт их.
        """

        rendered = [copy.deepcopy(message) for message in self.messages]
        for message in rendered:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if isinstance(call, dict):
                    call.pop("_malformed_arguments", None)
        if self.expected_tool_call_ids <= self.seen_tool_call_ids:
            rendered.extend(copy.deepcopy(self.pending_followups))
        return rendered
