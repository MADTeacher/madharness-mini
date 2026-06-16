"""Собираем сообщения, которые будут отправлены модели."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from typing import Any

from .budget import (
    TOKEN_ESTIMATE_BYTES_PER_TOKEN,
    clip_tool_messages,
    clip_text,
    estimate_request_tokens,
    estimate_tokens,
)
from .fragments import ContextFragment, ContextProvider, ContextState
from .history import HistoryEntry
from .render import render_messages

# Полный assistant-текст полезен только до разумного предела: модель уже
# получила свои рассуждения в прошлом ходе, а следующий запрос платит за них снова.
ASSISTANT_CONTENT_LIMIT = 8000

# Если assistant сразу вызывает tool, его текст обычно служебный; сохраняем кратко.
ASSISTANT_TOOL_CONTENT_LIMIT = 2000


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
        max_tokens: int = 60000,
        keep_recent_turns: int = 3,
        providers: Iterable[ContextProvider] | None = None,
    ):
        self.user_task = user_task
        self.max_tokens = max(int(max_tokens), 0)
        self.keep_recent_turns = max(int(keep_recent_turns), 0)
        self.providers = list(providers or [])
        self._fragments: list[ContextFragment] = []
        self._history: list[HistoryEntry] = []
        self._last_stats: dict[str, int | bool] | None = None
        self._last_report: dict[str, Any] | None = None

    def add_fragment(self, fragment: ContextFragment) -> None:
        """Добавляем или заменяем фрагмент по id."""

        self._fragments = [item for item in self._fragments if item.id != fragment.id]
        self._fragments.append(fragment)
        self._last_stats = None
        self._last_report = None

    def record_assistant(self, message: dict[str, Any]) -> None:
        """Запоминаем ответ модели как следующий атомарный элемент истории."""

        stored = _sanitize_assistant_message(message)
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
        self._last_report = None

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
        self._last_report = None

    def messages(self, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Возвращаем сообщения для модели с учётом бюджета контекста."""

        fragments = self._collect_fragments()
        entries = copy.deepcopy(self._history)
        entry_indexes = list(range(len(entries)))
        messages = render_messages(self.user_task, fragments, entries)
        initial_estimate = estimate_request_tokens(messages, tools)
        initial_tokens = initial_estimate["request_tokens_estimate"]
        truncated = False
        dropped_entries: list[dict[str, Any]] = []
        clipped_tool_messages: list[dict[str, Any]] = []
        clip_limit_chars = 0

        if self.max_tokens and initial_tokens > self.max_tokens:
            clip_limit_chars = max(
                80,
                min(4000, self.max_tokens * TOKEN_ESTIMATE_BYTES_PER_TOKEN // 8),
            )
            clipped_tool_messages = clip_tool_messages(entries, clip_limit_chars)
            if clipped_tool_messages:
                truncated = True
                messages = render_messages(self.user_task, fragments, entries)

        current_estimate = estimate_request_tokens(messages, tools)
        if (
            self.max_tokens
            and current_estimate["request_tokens_estimate"] > self.max_tokens
        ):
            dropped_entries = self._drop_old_entries_until_budget(
                fragments,
                entries,
                entry_indexes,
                tools,
            )
            messages = render_messages(self.user_task, fragments, entries)
            truncated = truncated or bool(dropped_entries)
            current_estimate = estimate_request_tokens(messages, tools)

        if (
            self.max_tokens
            and current_estimate["request_tokens_estimate"] > self.max_tokens
        ):
            forced_dropped = self._drop_old_entries_until_budget(
                fragments,
                entries,
                entry_indexes,
                tools,
                keep_recent_turns=0,
                forced=True,
            )
            dropped_entries.extend(forced_dropped)
            messages = render_messages(self.user_task, fragments, entries)
            truncated = truncated or bool(forced_dropped)
            current_estimate = estimate_request_tokens(messages, tools)

        hard_limit_exceeded = bool(
            self.max_tokens
            and current_estimate["request_tokens_estimate"] > self.max_tokens
        )
        context_packet = _context_packet_report(
            self.user_task,
            fragments,
            entries,
            entry_indexes,
            tools,
            current_estimate,
        )
        self._last_stats = {
            "context_tokens_estimate": current_estimate["request_tokens_estimate"],
            "messages_tokens_estimate": current_estimate["messages_tokens_estimate"],
            "tools_tokens_estimate": current_estimate["tools_tokens_estimate"],
            "fragments": len(fragments),
            "history_entries": len(self._history),
            "dropped_entries": len(dropped_entries),
            "truncated": truncated,
            "hard_limit_exceeded": hard_limit_exceeded,
        }
        self._last_report = {
            "max_tokens": self.max_tokens,
            "initial_request_tokens_estimate": initial_tokens,
            **current_estimate,
            "over_budget": bool(self.max_tokens and initial_tokens > self.max_tokens),
            "truncated": truncated,
            "hard_limit_exceeded": hard_limit_exceeded,
            "fragments": [_fragment_report(fragment) for fragment in fragments],
            "context_packet": context_packet,
            "history": {
                "total_entries": len(self._history),
                "rendered_entries": len(entries),
                "keep_recent_turns": self.keep_recent_turns,
                "clip_limit_chars": clip_limit_chars,
                "clipped_tool_messages": clipped_tool_messages,
                "dropped_entries": dropped_entries,
                "included_entries": [
                    _history_entry_report(entry, index)
                    for index, entry in zip(entry_indexes, entries)
                ],
            },
        }
        if hard_limit_exceeded:
            raise RuntimeError(
                "context budget exceeded after truncation: "
                f"{current_estimate['request_tokens_estimate']}/{self.max_tokens} "
                "estimated tokens"
            )
        return messages

    def stats(self) -> dict[str, int | bool]:
        """Короткая диагностика последней сборки контекста."""

        if self._last_stats is None:
            self.messages()
        return dict(self._last_stats or {})

    def report(self) -> dict[str, Any]:
        """Подробно описываем последнюю сборку контекста без текстов сообщений.

        Отчёт нужен для трасс и отладки бюджета: он показывает размеры,
        фрагменты, оставшуюся историю и действия обрезки, но не дублирует
        содержимое prompt/tool output.
        """

        if self._last_report is None:
            self.messages()
        return copy.deepcopy(self._last_report or {})

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
            max_tokens=self.max_tokens,
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
        entry_indexes: list[int],
        tools: list[dict[str, Any]] | None,
        keep_recent_turns: int | None = None,
        forced: bool = False,
    ) -> list[dict[str, Any]]:
        """Удаляем старые неприкреплённые элементы, сохраняя недавнюю историю."""

        dropped: list[dict[str, Any]] = []
        messages = render_messages(self.user_task, fragments, entries)
        keep_recent = (
            self.keep_recent_turns if keep_recent_turns is None else keep_recent_turns
        )
        protected_start = max(len(entries) - keep_recent, 0)
        while (
            entries
            and estimate_request_tokens(messages, tools)["request_tokens_estimate"]
            > self.max_tokens
        ):
            removable = next(
                (index for index in range(protected_start) if entries[index]),
                None,
            )
            if removable is None:
                break
            report = _history_entry_report(entries[removable], entry_indexes[removable])
            if forced:
                report["forced"] = True
            dropped.append(report)
            del entries[removable]
            del entry_indexes[removable]
            protected_start = max(len(entries) - keep_recent, 0)
            messages = render_messages(self.user_task, fragments, entries)
        return dropped


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Сохраняем в историю только поля, которые нужны следующему Chat request."""

    tool_calls = _sanitize_tool_calls(message.get("tool_calls") or [])
    content = message.get("content")
    limit = ASSISTANT_TOOL_CONTENT_LIMIT if tool_calls else ASSISTANT_CONTENT_LIMIT
    stored: dict[str, Any] = {"role": "assistant"}
    if isinstance(content, str):
        stored["content"] = clip_text(content, limit)
    elif content is None:
        stored["content"] = None if tool_calls else ""
    else:
        stored["content"] = copy.deepcopy(content)
    if tool_calls:
        stored["tool_calls"] = tool_calls
    return stored


def _sanitize_tool_calls(calls: list[Any]) -> list[dict[str, Any]]:
    """Оставляем у tool call только OpenAI-compatible id, type и function."""

    sanitized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        sanitized.append(
            {
                "id": str(call.get("id") or name),
                "type": str(call.get("type") or "function"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return sanitized


def _fragment_report(fragment: ContextFragment) -> dict[str, Any]:
    """Описываем фрагмент контекста без самого текста."""

    return {
        "id": fragment.id,
        "source": fragment.source,
        "placement": fragment.placement,
        "priority": fragment.priority,
        "chars": len(fragment.text),
        "transient": fragment.transient,
        "empty": not bool(fragment.text.strip()),
    }


def _history_entry_report(entry: HistoryEntry, index: int) -> dict[str, Any]:
    """Описываем элемент истории так, чтобы трасса не раздувалась контентом."""

    rendered = entry.rendered_messages()
    return {
        "index": index,
        "kind": entry.kind,
        "messages": len(entry.messages),
        "rendered_messages": len(rendered),
        "tokens_estimate": estimate_tokens(rendered),
        "roles": [str(message.get("role") or "") for message in rendered],
        "tool_call_ids": sorted(entry.expected_tool_call_ids | entry.seen_tool_call_ids),
        "pending_followups": len(entry.pending_followups),
    }


def _context_packet_report(
    user_task: str,
    fragments: list[ContextFragment],
    entries: list[HistoryEntry],
    entry_indexes: list[int],
    tools: list[dict[str, Any]] | None,
    estimate: dict[str, int],
) -> dict[str, Any]:
    """Пишем индекс prompt-сборки для будущей тепловой карты без текста prompt."""

    units: list[dict[str, Any]] = []
    cursor = 0

    for fragment in fragments:
        cursor = _append_context_unit(
            units,
            cursor,
            unit_id=f"fragment:{fragment.id}",
            source_type=_fragment_source_type(fragment),
            source_name=fragment.source,
            source_ref=fragment.id,
            payload=fragment.text,
            included_because=f"{fragment.placement}_fragment",
            metadata={
                "placement": fragment.placement,
                "priority": fragment.priority,
                "transient": fragment.transient,
                "chars": len(fragment.text),
            },
            confidence=0.95,
        )

    cursor = _append_context_unit(
        units,
        cursor,
        unit_id="user_task",
        source_type="user_message",
        source_name="task",
        source_ref="user_task",
        payload=user_task,
        included_because="current_task",
        metadata={"chars": len(user_task)},
        confidence=1.0,
    )

    for entry, original_index in zip(entries, entry_indexes):
        for message_index, message in enumerate(entry.rendered_messages()):
            source_type = _message_source_type(message)
            role = str(message.get("role") or "")
            cursor = _append_context_unit(
                units,
                cursor,
                unit_id=f"history:{original_index}:{message_index}",
                source_type=source_type,
                source_name=role or entry.kind,
                source_ref=f"history[{original_index}].messages[{message_index}]",
                payload=message,
                included_because="rendered_history",
                metadata={
                    "history_index": original_index,
                    "history_kind": entry.kind,
                    "role": role,
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "tool_call_ids": sorted(
                        entry.expected_tool_call_ids | entry.seen_tool_call_ids
                    ),
                },
                confidence=0.9,
            )

    for tool_index, tool in enumerate(tools or []):
        function = tool.get("function") if isinstance(tool, dict) else {}
        name = ""
        if isinstance(function, dict):
            name = str(function.get("name") or "")
        cursor = _append_context_unit(
            units,
            cursor,
            unit_id=f"tool_schema:{name or tool_index}",
            source_type="tool_schema",
            source_name=name or "tool_schema",
            source_ref=f"tools[{tool_index}]",
            payload=tool,
            included_because="available_tool_schema",
            metadata={"tool_index": tool_index},
            confidence=0.85,
        )

    warnings: list[str] = []
    if units and cursor != estimate["request_tokens_estimate"]:
        warnings.append("token_positions_are_estimates")
    return {
        "version": 1,
        "token_count_method": "char_estimate",
        "position_method": "estimated_sequential_units",
        "messages_tokens_estimate": estimate["messages_tokens_estimate"],
        "tools_tokens_estimate": estimate["tools_tokens_estimate"],
        "request_tokens_estimate": estimate["request_tokens_estimate"],
        "units": units,
        "warnings": warnings,
    }


def _append_context_unit(
    units: list[dict[str, Any]],
    cursor: int,
    *,
    unit_id: str,
    source_type: str,
    source_name: str,
    source_ref: str,
    payload: Any,
    included_because: str,
    metadata: dict[str, Any],
    confidence: float,
) -> int:
    """Добавляем один элемент prompt-индекса и возвращаем следующую позицию."""

    tokens = estimate_tokens(payload)
    end = cursor + tokens
    units.append(
        {
            "unit_id": unit_id,
            "source_type": source_type,
            "source_name": source_name,
            "source_ref": source_ref,
            "tokens_estimate": tokens,
            "position_start": cursor,
            "position_end": end,
            "included_because": included_because,
            "content_hash": _content_hash(payload),
            "confidence": confidence,
            "metadata": metadata,
        }
    )
    return end


def _content_hash(payload: Any) -> str:
    """Хэшируем содержимое, не записывая сам текст в trace."""

    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def _fragment_source_type(fragment: ContextFragment) -> str:
    """Грубо классифицируем закреплённые фрагменты для аналитики контекста."""

    value = f"{fragment.id} {fragment.source}".lower()
    if "system" in value:
        return "system_instruction"
    if "agents" in value or "project" in value or "instruction" in value:
        return "developer_instruction"
    if "skill" in value:
        return "developer_instruction"
    return "context_fragment"


def _message_source_type(message: dict[str, Any]) -> str:
    """Преобразуем Chat role в тип фрагмента тепловой карты."""

    role = str(message.get("role") or "")
    if role == "assistant":
        return "assistant_message"
    if role == "tool":
        return "tool_output"
    if role == "user":
        return "user_message"
    if role == "system":
        return "system_instruction"
    return "unknown"


def _tool_name(call: dict[str, Any], observation: dict[str, Any]) -> str:
    """Достаём имя инструмента для fallback tool_call_id."""

    function = call.get("function") or {}
    return str(function.get("name") or observation.get("tool") or "tool_call")
