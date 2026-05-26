"""Собираем сообщения, которые будут отправлены модели."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

from .budget import clip_tool_messages, messages_chars
from .fragments import ContextFragment, ContextProvider, ContextState
from .history import HistoryEntry
from .render import render_messages


class ContextManager:
    """Хранит контекст одного ask/run и собирает сообщения для Chat Completions.

    Менеджер ничего не знает о Config, Policy, ModelClient и реальных handlers.
    Loop сообщает ему факты: стартовые фрагменты, ответ модели и результат
    инструмента. На выходе получается обычный список сообщений для API.
    """

    def __init__(
        self,
        user_task: str,
        *,
        max_chars: int = 120000,
        keep_recent_turns: int = 3,
        providers: Iterable[ContextProvider] | None = None,
    ):
        self.user_task = user_task
        self.max_chars = max(int(max_chars), 0)
        self.keep_recent_turns = max(int(keep_recent_turns), 0)
        self.providers = list(providers or [])
        self._fragments: list[ContextFragment] = []
        self._history: list[HistoryEntry] = []
        self._last_stats: dict[str, int | bool] | None = None

    def add_fragment(self, fragment: ContextFragment) -> None:
        """Добавляем или заменяем фрагмент по id."""

        self._fragments = [item for item in self._fragments if item.id != fragment.id]
        self._fragments.append(fragment)
        self._last_stats = None

    def record_assistant(self, message: dict[str, Any]) -> None:
        """Запоминаем ответ модели как следующий атомарный элемент истории."""

        stored = copy.deepcopy(message)
        stored.setdefault("role", "assistant")
        expected = {
            str(call.get("id"))
            for call in stored.get("tool_calls") or []
            if call.get("id")
        }
        kind = "tool_turn" if expected else "assistant"
        self._history.append(
            HistoryEntry(
                kind=kind,
                messages=[stored],
                expected_tool_call_ids=expected,
            )
        )
        self._last_stats = None

    def record_tool_result(
        self,
        call: dict[str, Any],
        observation: dict[str, Any],
        followup_messages: Iterable[dict[str, Any]] = (),
    ) -> None:
        """Добавляем role=tool и отложенные follow-up сообщения."""

        entry = self._last_tool_entry()
        call_id = str(call.get("id") or _tool_name(call, observation))
        entry.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(observation, ensure_ascii=False),
            }
        )
        entry.seen_tool_call_ids.add(call_id)
        entry.pending_followups.extend(copy.deepcopy(list(followup_messages)))
        self._last_stats = None

    def messages(self) -> list[dict[str, Any]]:
        """Возвращаем сообщения для модели с учётом бюджета контекста."""

        fragments = self._collect_fragments()
        entries = copy.deepcopy(self._history)
        messages = render_messages(self.user_task, fragments, entries)
        truncated = False
        dropped_entries = 0

        if self.max_chars and messages_chars(messages) > self.max_chars:
            clip_limit = max(80, min(4000, self.max_chars // 8))
            if clip_tool_messages(entries, clip_limit):
                truncated = True
                messages = render_messages(self.user_task, fragments, entries)

        if self.max_chars and messages_chars(messages) > self.max_chars:
            dropped_entries = self._drop_old_entries_until_budget(fragments, entries)
            messages = render_messages(self.user_task, fragments, entries)
            truncated = truncated or dropped_entries > 0

        self._last_stats = {
            "context_chars": messages_chars(messages),
            "fragments": len(fragments),
            "history_entries": len(self._history),
            "dropped_entries": dropped_entries,
            "truncated": truncated,
        }
        return messages

    def stats(self) -> dict[str, int | bool]:
        """Короткая диагностика последней сборки контекста."""

        if self._last_stats is None:
            self.messages()
        return dict(self._last_stats or {})

    def _last_tool_entry(self) -> HistoryEntry:
        """Находим последний tool turn или создаём защитный entry для сбоя."""

        if self._history and self._history[-1].kind == "tool_turn":
            return self._history[-1]
        entry = HistoryEntry(kind="tool_turn")
        self._history.append(entry)
        return entry

    def _collect_fragments(self) -> list[ContextFragment]:
        """Собираем закреплённые и provider-фрагменты в стабильном порядке."""

        state = ContextState(
            user_task=self.user_task,
            fragments_count=len(self._fragments),
            history_entries=len(self._history),
            max_chars=self.max_chars,
            keep_recent_turns=self.keep_recent_turns,
        )
        fragments = list(self._fragments)
        for provider in self.providers:
            fragments.extend(provider.collect(state))
        return sorted(
            fragments,
            key=lambda item: (item.placement, item.priority, item.id),
        )

    def _drop_old_entries_until_budget(
        self,
        fragments: list[ContextFragment],
        entries: list[HistoryEntry],
    ) -> int:
        """Удаляем старые неприкреплённые элементы, сохраняя недавнюю историю."""

        dropped = 0
        messages = render_messages(self.user_task, fragments, entries)
        protected_start = max(len(entries) - self.keep_recent_turns, 0)
        while entries and messages_chars(messages) > self.max_chars:
            removable = next(
                (index for index in range(protected_start) if entries[index]),
                None,
            )
            if removable is None:
                break
            del entries[removable]
            dropped += 1
            protected_start = max(len(entries) - self.keep_recent_turns, 0)
            messages = render_messages(self.user_task, fragments, entries)
        return dropped


def _tool_name(call: dict[str, Any], observation: dict[str, Any]) -> str:
    """Достаём имя инструмента для fallback tool_call_id."""

    function = call.get("function") or {}
    return str(function.get("name") or observation.get("tool") or "tool_call")
