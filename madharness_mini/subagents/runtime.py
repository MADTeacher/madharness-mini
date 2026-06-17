"""Сводки локальных trace-файлов субагентов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import paths_from_patch


def trace_path_for_observation(path: Path, root: Path) -> str:
    """Показываем путь trace относительно workspace/cwd, если возможно."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def summarize_subagent_trace(path: Path) -> dict[str, Any]:
    """Собираем короткую сводку локальной трассы субагента для родителя."""

    events = _read_trace_events(path)
    tools = [event for event in events if event.get("event") == "tool_observation"]
    changed = changed_files_from_events(events)
    model_calls = [
        event for event in events if event.get("event") == "model_call_finished"
    ]
    return {
        "tool_calls": len(tools),
        "model_calls": len(model_calls),
        "changed_files": changed,
        "summary": (
            f"{len(model_calls)} model calls, "
            f"{len(tools)} tool calls, "
            f"{len(changed)} changed files"
        ),
    }


def changed_files_from_events(events: list[dict[str, Any]]) -> list[str]:
    """Достаём изменённые пути из args write_file/apply_patch в локальном trace."""

    changed: list[str] = []
    for event in events:
        if event.get("event") != "tool_observation":
            continue
        tool = event.get("tool")
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        if tool == "write_file" and isinstance(args.get("path"), str):
            changed.append(args["path"])
        if tool == "apply_patch" and isinstance(args.get("patch"), str):
            changed.extend(paths_from_patch(args["patch"]))
    return sorted(set(changed))


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    """Читаем JSONL trace без падения из-за одной битой строки."""

    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events
