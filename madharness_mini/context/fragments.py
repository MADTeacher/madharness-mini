"""Типы контекстных фрагментов и provider-интерфейс."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ContextFragment:
    """Кусок контекста, который нужно добавить к запросу модели.

    Закреплённые фрагменты вроде системного промпта и AGENTS.md живут отдельно
    от истории диалога: бюджет может сжимать старые tool outputs, но не должен
    случайно удалить правила проекта или исходную задачу.
    """

    id: str
    source: str
    text: str
    priority: int = 100
    placement: str = "system"
    transient: bool = False


@dataclass(frozen=True)
class ContextState:
    """Снимок состояния, на основе которого provider добавляет фрагменты.

    Provider получает только данные слоя контекста. Так будущие навыки, MCP или
    субагенты смогут подмешивать инструкции без импорта loop/model/tools.
    """

    user_task: str
    fragments_count: int
    history_entries: int
    max_tokens: int
    keep_recent_turns: int


class ContextProvider(Protocol):
    """Интерфейс расширения контекста для будущих источников."""

    def collect(self, state: ContextState) -> Iterable[ContextFragment]:
        """Возвращаем дополнительные фрагменты для текущего запроса."""
