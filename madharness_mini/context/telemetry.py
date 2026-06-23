"""Индекс prompt-сборки для трасс и тепловой карты контекста.

Модуль строит context_packet — структурированное описание того, из чего собран
запрос к модели: фрагменты, user_task (с отделением вложенных данных), история и
схемы tools. Каждой единице назначается классификация (authority_level,
context_layer, evictability и т.д.) и оценка позиций в токенах. Текстов сообщений
здесь нет — только хэши и метаданные, чтобы trace не раздувался и не сливал
содержимое prompt.

Внешних зависимостей от ContextManager нет: чистая функция build_context_packet
над уже собранными фрагментами и историей.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .budget import estimate_tokens
from .fragments import ContextFragment
from .history import HistoryEntry
from .sanitize import _tool_kind_and_path

# Маркер заменяет вложенный блок только в диагностическом goal-anchor hash.
ATTACHED_DATA_MARKER = "[attached data tracked as a separate context unit]"
# Fenced-блоки часто несут код, логи или вставленные документы из запроса.
FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Явные заголовки помогают отделить внешний материал от самой цели задачи.
ATTACHED_HEADING_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?"
    r"(?:attached data|attachment|external document|external text|document|log|"
    r"tool output|trace|web text|pasted text|вложенные данные|внешний документ|"
    r"документ|лог|вывод)\s*:?\s*$"
)


def build_context_packet(
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
            classification=_classification_from_fragment(fragment),
            confidence=0.95,
        )

    for unit in _user_task_units(user_task):
        cursor = _append_context_unit(units, cursor, **unit)

    for entry, original_index in zip(entries, entry_indexes):
        for message_index, message in enumerate(entry.rendered_messages()):
            source_type = _message_source_type(message)
            role = str(message.get("role") or "")
            metadata = {
                "history_index": original_index,
                "history_kind": entry.kind,
                "role": role,
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "tool_call_ids": sorted(
                    entry.expected_tool_call_ids | entry.seen_tool_call_ids
                ),
            }
            # Для tool-наблюдений поднимаем имя инструмента и путь из JSON-контента:
            # иначе heat-анализатор видит все чтения/выводы как безымянный
            # "tool_output" и не может отличить read_file, чтобы проверить свежесть
            # чтения перед правкой (ложный cold gap). Поля работают и после
            # возрастной эвикции — digest_read_file сохраняет tool и path.
            if role == "tool":
                content = message.get("content")
                if isinstance(content, str):
                    metadata.update(_tool_observation_meta(content))
            cursor = _append_context_unit(
                units,
                cursor,
                unit_id=f"history:{original_index}:{message_index}",
                source_type=source_type,
                source_name=role or entry.kind,
                source_ref=f"history[{original_index}].messages[{message_index}]",
                payload=message,
                included_because="rendered_history",
                metadata=metadata,
                classification=_classification_for_source_type(source_type),
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
            classification=_classification_for_source_type("tool_schema"),
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


def _classification_from_fragment(fragment: ContextFragment) -> dict[str, str]:
    """Берём явную классификацию фрагмента или fallback по source_type."""

    fallback = _classification_for_source_type(_fragment_source_type(fragment))
    explicit = {
        "authority_level": fragment.authority_level,
        "context_layer": fragment.context_layer,
        "evictability": fragment.evictability,
        "stability": fragment.stability,
        "applicability": fragment.applicability,
        "normative_role": fragment.normative_role,
        "goal_role": fragment.goal_role,
    }
    return {
        key: value if value != "unknown" and value != "none" else fallback[key]
        for key, value in explicit.items()
    }


def _classification_for_source_type(source_type: str) -> dict[str, str]:
    """Назначаем базовый слой для старых фрагментов без явной разметки."""

    if source_type == "system_instruction":
        return {
            "authority_level": "system",
            "context_layer": "normative",
            "evictability": "never",
            "stability": "stable",
            "applicability": "active",
            "normative_role": "safety",
            "goal_role": "none",
        }
    if source_type == "developer_instruction":
        return {
            "authority_level": "developer",
            "context_layer": "normative",
            "evictability": "never",
            "stability": "stable",
            "applicability": "active",
            "normative_role": "workflow",
            "goal_role": "none",
        }
    if source_type == "user_message":
        return {
            "authority_level": "user",
            "context_layer": "goal",
            "evictability": "goal_update_only",
            "stability": "task",
            "applicability": "active",
            "normative_role": "none",
            "goal_role": "primary_goal",
        }
    if source_type in {"file_snippet", "test_result"}:
        return {
            "authority_level": "tool",
            "context_layer": "evidence",
            "evictability": "normal",
            "stability": "turn",
            "applicability": "current_task",
            "normative_role": "none",
            "goal_role": "none",
        }
    if source_type == "tool_schema":
        return {
            "authority_level": "tool",
            "context_layer": "tooling",
            "evictability": "normal",
            "stability": "session",
            "applicability": "active",
            "normative_role": "none",
            "goal_role": "none",
        }
    if source_type == "assistant_message":
        return {
            "authority_level": "assistant",
            "context_layer": "working",
            "evictability": "preferred",
            "stability": "turn",
            "applicability": "current_task",
            "normative_role": "none",
            "goal_role": "none",
        }
    if source_type == "tool_output":
        return {
            "authority_level": "tool",
            "context_layer": "working",
            "evictability": "preferred",
            "stability": "turn",
            "applicability": "current_task",
            "normative_role": "none",
            "goal_role": "none",
        }
    return {
        "authority_level": "unknown",
        "context_layer": "unknown",
        "evictability": "normal",
        "stability": "unknown",
        "applicability": "unknown",
        "normative_role": "none",
        "goal_role": "none",
    }


def _user_task_units(user_task: str) -> list[dict[str, Any]]:
    """Разделяем активную цель и вложенные пользовательские данные в telemetry."""

    spans = _attached_data_spans(user_task)
    primary_parts: list[str] = []
    previous = 0
    for start, end in spans:
        primary_parts.append(user_task[previous:start])
        primary_parts.append(ATTACHED_DATA_MARKER)
        previous = end
    primary_parts.append(user_task[previous:])
    primary_payload = "".join(primary_parts).strip() or "user provided attached data"
    attached_chars = sum(end - start for start, end in spans)
    units: list[dict[str, Any]] = [
        {
            "unit_id": "user_task",
            "source_type": "user_message",
            "source_name": "task",
            "source_ref": "user_task",
            "payload": primary_payload,
            "included_because": "current_task",
            "metadata": {
                "chars": len(user_task),
                "goal_anchor_chars": len(primary_payload),
                "attached_data_units": len(spans),
                "attached_data_chars": attached_chars,
            },
            "classification": {
                "authority_level": "user",
                "context_layer": "goal",
                "evictability": "goal_update_only",
                "stability": "task",
                "applicability": "active",
                "normative_role": "none",
                "goal_role": "primary_goal",
            },
            "confidence": 1.0,
        }
    ]
    for index, (start, end) in enumerate(spans):
        payload = user_task[start:end]
        taint_score = _attached_data_taint_score(payload)
        units.append(
            {
                "unit_id": f"user_task_attached:{index}",
                "source_type": "user_message",
                "source_name": "attached_data",
                "source_ref": f"user_task.attached[{index}]",
                "payload": payload,
                "included_because": "user_attached_data",
                "metadata": {
                    "chars": len(payload),
                    "attached_data_index": index,
                    "attached_data_taint_score": round(taint_score, 4),
                    "attachment_kind": _attached_data_kind(payload),
                },
                "classification": {
                    "authority_level": "external" if taint_score >= 0.50 else "user",
                    "context_layer": _attached_data_context_layer(payload),
                    "evictability": "preferred",
                    "stability": "temporary",
                    "applicability": "current_task",
                    "normative_role": "none",
                    "goal_role": "attached_data",
                },
                "confidence": 0.85,
            }
        )
    return units


def _attached_data_spans(text: str) -> list[tuple[int, int]]:
    """Находим вложенные блоки, которые не должны становиться goal anchor."""

    spans = [(match.start(), match.end()) for match in FENCED_BLOCK_RE.finditer(text)]
    heading_matches = list(ATTACHED_HEADING_RE.finditer(text))
    section_heading = re.compile(r"(?m)^#{1,6}\s+\S")
    for match in heading_matches:
        start = match.start()
        if any(span_start <= start < span_end for span_start, span_end in spans):
            continue
        next_match = section_heading.search(text, match.end())
        end = next_match.start() if next_match else len(text)
        spans.append((start, end))
    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Склеиваем пересекающиеся диапазоны вложенных данных."""

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _attached_data_taint_score(text: str) -> float:
    """Оцениваем, похож ли вложенный материал на чужие инструкции."""

    lowered = text.lower()
    instruction_markers = (
        "ignore previous instructions",
        "system prompt",
        "developer message",
        "you are chatgpt",
        "follow these instructions",
        "do not obey",
        "не выполняй предыдущие инструкции",
        "системный промпт",
        "инструкция разработчика",
    )
    if any(marker in lowered for marker in instruction_markers):
        return 0.85
    if any(marker in lowered for marker in ("instruction", "инструкция", "prompt")):
        return 0.45
    return 0.0


def _attached_data_kind(text: str) -> str:
    """Помечаем тип вложения для отчета без сохранения самого содержимого."""

    lowered = text.lower()
    if any(marker in lowered for marker in ("trace", "stderr", "stdout", "log", "error")):
        return "log"
    if any(marker in lowered for marker in ("test", "pytest", "unittest", "failed")):
        return "test_output"
    return "external_text"


def _attached_data_context_layer(text: str) -> str:
    """Логи и тесты считаем evidence, остальные вложения — working."""

    if _attached_data_kind(text) in {"log", "test_output"}:
        return "evidence"
    return "working"


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
    classification: dict[str, str],
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
            **classification,
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


def _tool_observation_meta(content: str) -> dict[str, str]:
    """Метаданные tool-наблюдения для context_packet: имя инструмента и путь.

    Heat-анализатор читает metadata.tool_name и metadata.path, чтобы отличать
    read_file от прочих tool_output и сверять свежесть чтения перед правкой.
    Без этих полей все tool-наблюдения выглядят безымянными — анализатор ставит
    ложный cold gap. Безопасный пустой возврат, если JSON не разобрался или полей
    нет: метаданные просто не дополняются, поведение остаётся прежним.
    """

    tool_name, path_value = _tool_kind_and_path(content)
    meta: dict[str, str] = {}
    if tool_name:
        meta["tool_name"] = tool_name
    if path_value:
        meta["path"] = path_value
    return meta
