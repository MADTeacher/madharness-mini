"""Нормализация исходных trace в переносимый JSONL событий."""

from __future__ import annotations

from pathlib import Path

from .io import read_jsonl, write_jsonl
from .loaders.madharness_trace import load_trace_path
from .schema import SessionEvent


def normalize_input(path: Path, out_dir: Path) -> tuple[list[SessionEvent], list[dict]]:
    """Читает trace-файл или каталог и пишет `events.jsonl` с warnings."""

    events, warnings = load_trace_path(path)
    write_jsonl(out_dir / "events.jsonl", [event.to_dict() for event in events])
    write_jsonl(out_dir / "warnings.jsonl", warnings)
    return events, warnings


def load_normalized_events(path: Path) -> list[SessionEvent]:
    """Загружает `events.jsonl`, созданный командой normalize."""

    events: list[SessionEvent] = []
    for row in read_jsonl(path):
        events.append(
            SessionEvent(
                event_id=str(row["event_id"]),
                session_id=str(row["session_id"]),
                turn_id=int(row.get("turn_id") or 0),
                timestamp=row.get("timestamp"),
                event_type=str(row.get("event_type") or "custom_event"),
                actor=str(row.get("actor") or "harness"),
                payload=dict(row.get("payload") or {}),
                raw_ref=dict(row.get("raw_ref") or {}),
                confidence=float(row.get("confidence") or 0.0),
            )
        )
    return events
