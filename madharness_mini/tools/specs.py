"""Описание инструмента и его схемы для OpenAI-совместимого API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .context import ToolContext

ToolHandler = Callable[["ToolContext", dict[str, Any]], dict[str, Any]]


@dataclass
class ToolSpec:
    """Один инструмент: имя, описание для модели, JSON Schema и handler.

    Handler получает общий контекст инструментов и аргументы конкретного вызова.
    Так новый инструмент можно держать в отдельном модуле, не раздувая registry.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        """Упаковываем инструмент в вид, который ждёт Chat Completions API."""

        function = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        return {"type": "function", "function": function}
