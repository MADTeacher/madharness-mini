"""Сбор данных «анатомии сессии» для heatmap-рендеров.

Общий агрегатор для PNG и HTML: оба рендера получают один и тот же dict, поэтому
логику подсчёта не нужно дублировать, а тестировать можно данные, а не пиксели.
Здесь нет рисования — только индексные, безопасные агрегаты (токены, доли,
номера ходов, имена инструментов). Полный текст запроса и ответы инструментов
не попадают в выход, как и во всём `context_heatmap`.
"""

from __future__ import annotations

import math
from typing import Any

from .png import SOURCE_ORDER


# Пороги взяты из docs/context-heatmap.md, раздел «Пороговые значения».
# Используются и PNG, и HTML для калибровки глаз и verdict-логики.
THRESHOLDS = {
    "red_token_share": 0.25,
    "cold_gap_score": 0.50,
    "assistant_share": 0.40,
    "window_pressure_score": 0.0001,
    "fill_warning": 0.75,
    "fill_danger": 0.90,
}

# Порядок рисования действий в action track. Имена инструментов — из builtin
# tools harness: read_file/write_file/apply_patch/list_files/run_shell/test*.
ACTION_GROUPS = {
    "read": {"read_file"},
    "write": {"write_file", "apply_patch"},
    "test": {"run_tests", "test"},
    "list": {"list_files"},
    "run": {"run_shell", "run_shell_background"},
}


def build_anatomy_data(
    report: dict[str, Any],
    packets: list[dict[str, Any]],
    turn_heat: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Собирает индексные агрегаты для heatmap «анатомии сессии».

    Аргументы — уже десериализованные dict'и артефактов сессии: `packets`
    (packets.jsonl), `turn_heat` (turn_heat.jsonl), `findings` (findings.jsonl),
    `events` (events.jsonl). Возвращает dict, понятный обоим рендерам.
    """

    columns = _build_columns(packets, turn_heat)
    cold_turns = _cold_turns(findings)
    actions_by_turn = _actions_by_turn(events)
    seams = _summarization_seams(events)
    verdict = _verdict(report, turn_heat, cold_turns)

    return {
        "session_id": str(report.get("session_id") or "session"),
        "columns": columns,
        "source_types": list(SOURCE_ORDER),
        "cold_turns": cold_turns,
        "actions_by_turn": actions_by_turn,
        "action_groups": {group: list(names) for group, names in ACTION_GROUPS.items()},
        "seams": seams,
        "verdict": verdict,
        "thresholds": THRESHOLDS,
        "metrics": _header_metrics(report, columns),
    }


def _build_columns(
    packets: list[dict[str, Any]],
    turn_heat: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Сводим каждый prompt-пакет к одной колонке «анатомии».

    Колонка хранит доли типов в окне (блок A) и значения сигналов по ходу
    (блок B): cold/pressure/red/fill. Размеры берём из tokens фрагментов и
    input_tokens пакета; дубликаты по model_call_id с turn_heat — основной
    способ привязки сигналов к колонке.
    """

    heat_by_call = {str(row.get("model_call_id") or ""): row for row in turn_heat}
    heat_by_turn = {_int(row.get("turn_id")): row for row in turn_heat}
    columns: list[dict[str, Any]] = []

    for index, packet in enumerate(packets):
        model_call_id = str(packet.get("model_call_id") or "")
        turn_id = _int(packet.get("turn_id"), index)
        heat = heat_by_call.get(model_call_id) or heat_by_turn.get(turn_id) or {}

        # Считаем токены по source_type так же, как старый png.py.
        tokens_by_type: dict[str, int] = {key: 0 for key in SOURCE_ORDER}
        fragment_count_by_type: dict[str, int] = {key: 0 for key in SOURCE_ORDER}
        for fragment in packet.get("fragments") or []:
            if not isinstance(fragment, dict):
                continue
            source_type = _source_type(str(fragment.get("source_type") or "unknown"))
            tokens = max(_int(fragment.get("tokens")), 0)
            tokens_by_type[source_type] = tokens_by_type.get(source_type, 0) + tokens
            fragment_count_by_type[source_type] += 1

        fragment_tokens = sum(tokens_by_type.values())
        input_tokens = max(_int(packet.get("input_tokens")), 0)
        window_tokens = _int(packet.get("context_window_tokens"))
        if window_tokens <= 0:
            window_tokens = max(input_tokens, fragment_tokens, 1)
        used_tokens = min(max(input_tokens or fragment_tokens, fragment_tokens), window_tokens)
        # Неучтённые токены (input есть, а во фрагментах нет) относим к unknown,
        # чтобы высота заливки честно отражала заполненность окна.
        if used_tokens > fragment_tokens:
            tokens_by_type["unknown"] += used_tokens - fragment_tokens

        fill_share = round(used_tokens / max(window_tokens, 1), 4)
        columns.append(
            {
                "turn_id": turn_id,
                "model_call_id": model_call_id,
                "window_tokens": max(window_tokens, 1),
                "used_tokens": max(min(used_tokens, window_tokens), 0),
                "fill_share": fill_share,
                "tokens_by_type": tokens_by_type,
                "fragment_count_by_type": fragment_count_by_type,
                "cold_gap_score": _float(heat.get("cold_gap_score")),
                "window_pressure_score": _float(heat.get("window_pressure_score")),
                "assistant_share": _float(heat.get("assistant_share")),
                "red_token_share": _float(heat.get("red_token_share")),
                "raw_tool_share": _float(heat.get("raw_tool_share")),
            }
        )
    return columns


def _cold_turns(findings: list[dict[str, Any]]) -> list[int]:
    """Собираем номера ходов, где сработал cold gap, без дублей."""

    seen: set[int] = set()
    ordered: list[int] = []
    for finding in findings:
        if str(finding.get("kind") or "") != "cold_gap":
            continue
        turn_id = _int(finding.get("turn_id"))
        if turn_id not in seen:
            seen.add(turn_id)
            ordered.append(turn_id)
    return ordered


def _actions_by_turn(events: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Имена вызванных инструментов по ходам — для action track.

    Берём из model_output: payload.turn + payload.message.tool_calls[].name.
    Дубликаты имён на одном ходе схлопываем, сохраняя порядок первого вызова.
    """

    actions: dict[int, list[str]] = {}
    for event in events:
        if str(event.get("event_type") or "") != "model_output":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        turn_id = _int(payload.get("turn"))
        calls = ((payload.get("message") or {}).get("tool_calls")) if isinstance(
            payload.get("message"), dict
        ) else None
        if not isinstance(calls, list):
            continue
        bucket = actions.setdefault(turn_id, [])
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name and name not in bucket:
                bucket.append(name)
    return actions


def _summarization_seams(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Швы summarization по ходам — где harness свернул или обрезал историю.

    Признаки лежат в payload.context_report.history ближайшего model_call:
    summarized_old_entries (свернуто в digest) и dropped_entries/clipped
    (аварийное усечение). Возвращаем только активные швы (где что-то свёрнуто),
    чтобы PNG/HTML не рисовали лишние линии на здоровых ходах.
    """

    seams: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("event_type") or "") != "model_call":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        report = payload.get("context_report")
        if not isinstance(report, dict):
            continue
        history = report.get("history")
        if not isinstance(history, dict):
            continue
        summarized = _len(history.get("summarized_old_entries"))
        dropped = _len(history.get("dropped_entries"))
        clipped = _len(history.get("clipped_tool_messages"))
        truncated = bool(report.get("truncated")) or dropped > 0
        if summarized == 0 and not truncated and clipped == 0:
            continue
        turn_id = _int(payload.get("turn"))
        seams.append(
            {
                "turn_id": turn_id,
                "summarized": summarized,
                "dropped": dropped,
                "clipped": clipped,
                "truncated": truncated,
            }
        )
    return seams


def _verdict(
    report: dict[str, Any],
    turn_heat: list[dict[str, Any]],
    cold_turns: list[int],
) -> dict[str, Any]:
    """Сводим состояние сессии в «светофор» и указатель на пиковый ход.

    Три точки — red heat / cold gaps / window pressure. Каждая загорается по
    порогу из THRESHOLDS. peak_turn — ход с максимумом главного сигнала
    (cold → red → pressure по приоритету), чтобы verdict работал указателем,
    а не лампочкой.
    """

    max_red = _float(report.get("max_red_token_share"))
    max_cold = _float(report.get("max_cold_gap_score"))
    max_pressure = _float(report.get("max_window_pressure_score"))
    max_assistant = _float(report.get("max_assistant_share"))

    red_on = max_red >= THRESHOLDS["red_token_share"]
    cold_on = max_cold >= THRESHOLDS["cold_gap_score"] or bool(cold_turns)
    pressure_on = (
        max_pressure > THRESHOLDS["window_pressure_score"]
        or max_assistant >= THRESHOLDS["assistant_share"]
    )
    dots = (
        ("1" if red_on else "0")
        + ("1" if cold_on else "0")
        + ("1" if pressure_on else "0")
    )

    parts = []
    if red_on:
        parts.append("RED")
    if cold_on:
        parts.append("COLD")
    if pressure_on:
        parts.append("PRESSURE")
    label = "+".join(parts) if parts else "CLEAN"

    peak_turn = _peak_turn(turn_heat)
    hint = _verdict_hint(red_on, cold_on, pressure_on)
    return {
        "dots": dots,
        "label": label,
        "peak_turn": peak_turn,
        "hint": hint,
        "max_red_token_share": round(max_red, 4),
        "max_cold_gap_score": round(max_cold, 4),
        "max_window_pressure_score": round(max_pressure, 4),
        "max_assistant_share": round(max_assistant, 4),
    }


def _peak_turn(turn_heat: list[dict[str, Any]]) -> int | None:
    """Ход с самым высоким diagnostic-сигналом: сначала cold, потом red, потом pressure."""

    if not turn_heat:
        return None
    ranked = sorted(
        turn_heat,
        key=lambda row: (
            _float(row.get("cold_gap_score")),
            _float(row.get("red_token_share")),
            _float(row.get("window_pressure_score")),
        ),
        reverse=True,
    )
    best = ranked[0]
    if (
        _float(best.get("cold_gap_score")) <= 0.0
        and _float(best.get("red_token_share")) <= 0.0
        and _float(best.get("window_pressure_score")) <= 0.0
    ):
        return None
    return _int(best.get("turn_id"))


def _verdict_hint(red_on: bool, cold_on: bool, pressure_on: bool) -> str:
    """Куда смотреть дальше — превращаем verdict в инструкцию, а не лампочку."""

    if red_on:
        return "fragment_heat.jsonl / Context Window"
    if cold_on and pressure_on:
        return "findings.jsonl + summarization policy"
    if cold_on:
        return "findings.jsonl"
    if pressure_on:
        return "summarization policy (assistant_share)"
    return "no action needed"


def _header_metrics(
    report: dict[str, Any],
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Числа для строки заголовка: сессия, ходы, findings, max fill."""

    max_fill = max((col["fill_share"] for col in columns), default=0.0)
    return {
        "model_calls": int(report.get("model_calls") or len(columns)),
        "turns": int(report.get("turns") or len(columns)),
        "findings": int(report.get("findings") or 0),
        "warnings": int(report.get("warnings") or 0),
        "max_fill": round(max_fill, 4),
        "max_fixed_instruction_cost": round(
            _float(report.get("max_fixed_instruction_cost")), 4
        ),
        "max_goal_anchor_cost": round(_float(report.get("max_goal_anchor_cost")), 4),
    }


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _len(value: Any) -> int:
    """Длина list-поля из trace без падения на не-списках."""

    if isinstance(value, list):
        return len(value)
    return 0


def _source_type(value: str) -> str:
    return value if value in SOURCE_ORDER else "unknown"
