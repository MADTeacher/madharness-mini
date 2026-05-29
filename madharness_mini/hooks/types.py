"""Типы публичного контракта пользовательских hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

HOOK_SCHEMA_VERSION = 1

HOOK_EVENTS = {
    "session_start",
    "before_model_call",
    "after_model_call",
    "before_tool_call",
    "after_tool_call",
    "session_end",
    "session_error",
}


@dataclass(frozen=True)
class HookEvent:
    """Одно событие харнесса, которое пользовательский hook получает в stdin."""

    name: str
    kind: str
    trace_id: str
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Передаём hook-скрипту стабильный JSON-контракт события."""

        return {
            "version": HOOK_SCHEMA_VERSION,
            "event": self.name,
            "kind": self.kind,
            "trace_id": self.trace_id,
            "data": self.data,
        }


@dataclass(frozen=True)
class HookDecision:
    """Решение hook: продолжить выполнение или заблокировать действие."""

    ok: bool = True
    block: str = ""
    message: str = ""


class HookProvider(Protocol):
    """Минимальный provider, который обрабатывает одно событие hooks."""

    id: str

    def matches(self, event: HookEvent) -> bool:
        """Проверяем, должен ли provider увидеть это событие."""

        ...

    def handle(self, event: HookEvent) -> HookDecision:
        """Выполняем hook и возвращаем решение для manager-а."""

        ...
