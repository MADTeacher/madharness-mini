"""Загрузка JSONL-трасс `madharness-mini` в нормализованные события."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import SessionEvent


EVENT_TYPE_MAP = {
    "session_start": "session_start",
    "model_call_started": "model_call",
    "model_call_finished": "model_output",
    "tool_observation": "tool_result",
    "context_error": "context_error",
    "session_end": "session_end",
    "skill_activated": "memory_write",
    "subagent_started": "branch_start",
    "subagent_finished": "branch_end",
    "user_input_requested": "permission_event",
}


def load_trace(path: Path) -> tuple[list[SessionEvent], list[dict[str, Any]]]:
    """Читаем один trace-файл и возвращаем события плюс предупреждения."""

    session_id = path.stem
    events: list[SessionEvent] = []
    warnings: list[dict[str, Any]] = []
    current_turn = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(
                {
                    "kind": "invalid_jsonl",
                    "file": str(path),
                    "line": line_number,
                    "message": str(exc),
                }
            )
            continue
        if not isinstance(raw, dict):
            warnings.append(
                {
                    "kind": "non_object_event",
                    "file": str(path),
                    "line": line_number,
                }
            )
            continue
        raw_event = str(raw.get("event") or "custom_event")
        if isinstance(raw.get("turn"), int):
            current_turn = int(raw["turn"])
        event_id = f"{session_id}:evt-{line_number:06d}"
        events.append(
            SessionEvent(
                event_id=event_id,
                session_id=session_id,
                turn_id=current_turn,
                timestamp=_optional_float(raw.get("ts")),
                event_type=EVENT_TYPE_MAP.get(raw_event, "custom_event"),
                actor=_actor_for(raw_event),
                payload={key: value for key, value in raw.items() if key != "ts"},
                raw_ref={"file": str(path), "line": line_number, "offset": None},
                confidence=1.0,
            )
        )
    if not any(event.event_type == "model_call" for event in events):
        warnings.append(
            {
                "kind": "missing_model_call",
                "file": str(path),
                "message": "trace does not contain model_call_started",
            }
        )
    return events, warnings


def load_trace_path(path: Path) -> tuple[list[SessionEvent], list[dict[str, Any]]]:
    """Читает один файл или каталог trace-файлов."""

    if path.is_dir():
        all_events: list[SessionEvent] = []
        all_warnings: list[dict[str, Any]] = []
        for item in sorted(path.glob("*.jsonl")):
            events, warnings = load_trace(item)
            all_events.extend(events)
            all_warnings.extend(warnings)
        return all_events, all_warnings
    return load_trace(path)


def _actor_for(raw_event: str) -> str:
    """Грубо назначаем автора события для общей схемы."""

    if raw_event.startswith("model_"):
        return "model"
    if raw_event == "tool_observation":
        return "tool"
    if raw_event in {"session_start", "session_end", "context_error"}:
        return "harness"
    return "agent"


def _optional_float(value: Any) -> float | None:
    """Приводим timestamp к float, если возможно."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
