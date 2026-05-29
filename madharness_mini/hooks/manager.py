"""Синхронный dispatcher пользовательских hooks."""

from __future__ import annotations

import time
from typing import Any

from ..config import Config
from ..policy import Policy
from ..trace import Trace
from ..utils import clipped
from .commands import CommandHookProvider
from .config import load_command_hook_configs
from .redaction import compact_payload
from .types import HookDecision, HookEvent, HookProvider


class HookManager:
    """Вызывает включённые hooks по порядку и пишет их результат в trace.

    Это намеренно маленький dispatcher, а не глобальный event bus: харнесс явно
    вызывает `emit()` в нескольких lifecycle-точках, а manager синхронно получает
    первое блокирующее решение.
    """

    def __init__(self, providers: list[HookProvider], trace: Trace | None = None):
        self.providers = providers
        self.trace = trace

    @classmethod
    def from_config(cls, cfg: Config, trace: Trace | None = None) -> "HookManager":
        """Строим manager из `.madharness-mini/hooks.json`; без файла будет no-op."""

        policy = Policy(cfg)
        providers = [
            CommandHookProvider(config)
            for config in load_command_hook_configs(cfg, policy)
        ]
        return cls(providers, trace)

    def with_trace(self, trace: Trace | None) -> "HookManager":
        """Переиспользуем те же hooks в дочернем trace, например у субагента."""

        return HookManager(list(self.providers), trace)

    def emit(
        self,
        name: str,
        *,
        kind: str,
        data: dict[str, Any] | None = None,
    ) -> HookDecision:
        """Сообщаем hooks о событии и возвращаем первое блокирующее решение."""

        if not self.providers:
            return HookDecision()
        event = HookEvent(
            name=name,
            kind=kind,
            trace_id=self.trace.id if self.trace else "",
            data=compact_payload(data or {}),
        )
        for provider in self.providers:
            if not provider.matches(event):
                continue
            decision = self._handle_provider(provider, event)
            if not decision.ok:
                return decision
        return HookDecision()

    def _handle_provider(
        self,
        provider: HookProvider,
        event: HookEvent,
    ) -> HookDecision:
        """Выполняем один hook и превращаем его сбои в trace, не ломая run."""

        started = time.monotonic()
        self._write(
            "hook_started",
            hook=provider.id,
            hook_event=event.name,
            kind=event.kind,
        )
        try:
            decision = provider.handle(event)
        except Exception as exc:
            self._write(
                "hook_failed",
                hook=provider.id,
                hook_event=event.name,
                kind=event.kind,
                error=clipped(str(exc), 1000),
                elapsed_ms=_elapsed_ms(started),
            )
            return HookDecision()
        if decision.ok:
            self._write(
                "hook_finished",
                hook=provider.id,
                hook_event=event.name,
                kind=event.kind,
                message=clipped(decision.message, 1000),
                elapsed_ms=_elapsed_ms(started),
            )
            return decision
        self._write(
            "hook_blocked",
            hook=provider.id,
            hook_event=event.name,
            kind=event.kind,
            block=clipped(decision.block, 1000),
            message=clipped(decision.message, 1000),
            elapsed_ms=_elapsed_ms(started),
        )
        return decision

    def _write(self, event: str, **data: Any) -> None:
        """Пишем hook-события в общий JSONL trace, если trace есть."""

        if self.trace:
            self.trace.write(event, **data)


def _elapsed_ms(started: float) -> int:
    """Показываем длительность hook без лишней точности."""

    return int((time.monotonic() - started) * 1000)
