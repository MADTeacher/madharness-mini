"""Описание инструмента и его схемы для OpenAI-совместимого API."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from .context import ToolContext

ToolHandler = Callable[["ToolContext", dict[str, Any]], dict[str, Any]]


class ToolProvider(Protocol):
    """Источник ToolSpec для встроенных и будущих внешних инструментов."""

    def specs(self, ctx: "ToolContext") -> Iterable["ToolSpec"]:
        """Возвращаем инструменты, доступные в данном tool context."""


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
