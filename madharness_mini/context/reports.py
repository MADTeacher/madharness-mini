"""Описания объектов контекста для trace без самих текстов.

Функции отсюда превращают фрагмент или элемент истории в словарь-описание для
отчёта/трассы: показывают размеры, роли, идентификаторы и файловые эффекты, но
не дублируют содержимое prompt и tool output. Это изоляция «человеческой»
диагностики от ядра сборки сообщений.
"""

from __future__ import annotations

from typing import Any

from .budget import estimate_tokens
from .fragments import ContextFragment
from .history import HistoryEntry


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
        "authority_level": fragment.authority_level,
        "context_layer": fragment.context_layer,
        "evictability": fragment.evictability,
        "stability": fragment.stability,
        "applicability": fragment.applicability,
        "normative_role": fragment.normative_role,
        "goal_role": fragment.goal_role,
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
        # Файловые эффекты без хэшей содержимого: для диагностики дедупа и
        # напоминания о состоянии достаточно путей и типов воздействия.
        "file_refs": [{"path": ref.path, "kind": ref.kind} for ref in entry.file_refs],
    }
