"""JSONL-трассы запусков ask/run и команда trace для просмотра."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import Config


class Trace:
    """Один файл трассы на сессию: события по мере работы ask или run."""

    def __init__(
        self,
        cfg: Config,
        kind: str,
        *,
        parent_id: str | None = None,
        label: str | None = None,
    ):
        cfg.ensure_dirs()
        self.cfg = cfg
        suffix = uuid.uuid4().hex[:8]
        if parent_id and label:
            self.id = f"{parent_id}--{_safe_trace_label(label)}-{suffix}"
        else:
            self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
        self.path = cfg.state_dir.joinpath("traces", f"{self.id}.jsonl")
        data: dict[str, Any] = {"kind": kind}
        if parent_id:
            data["parent_id"] = parent_id
        if label:
            data["label"] = label
        self.write("session_start", **data)

    def write(self, event: str, **data: Any) -> None:
        """Дописываем одно событие в конец JSONL (модель, tool, конец сессии)."""

        record = {"ts": time.time(), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def child(self, kind: str, label: str) -> "Trace":
        """Создаём локальную трассу дочернего запуска и связываем её с parent id."""

        return Trace(self.cfg, kind, parent_id=self.id, label=label)


def summarize_trace(cfg: Config, trace_id: str) -> str:
    """Текстовая сводка для CLI: путь, число событий, tool calls, итог.

    trace_id может быть префиксом имени файла в каталоге traces/.
    """

    path = _resolve_trace_path(cfg, trace_id)
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tools = [e for e in events if e.get("event") == "tool_observation"]
    discovered = [
        e for e in events if e.get("event") == "skills_discovered"
    ]
    activated = [
        e for e in events if e.get("event") == "skill_activated"
    ]
    resources = [
        e for e in events if e.get("event") == "skill_resource_used"
    ]
    reports = [
        e["context_report"]
        for e in events
        if isinstance(e.get("context_report"), dict)
    ]
    end = next((e for e in reversed(events) if e.get("event") == "session_end"), {})
    result = str(end.get("result", ""))[:1000]
    lines = [
        f"trace: {path}",
        f"events: {len(events)}",
        f"tool calls: {len(tools)}",
    ]
    if reports:
        lines.append(_summarize_context_report(reports[-1]))
    if discovered or activated or resources:
        last_discovered = discovered[-1] if discovered else {}
        activated_names = sorted({str(event.get("name")) for event in activated})
        lines.append(
            "skills: "
            f"discovered {int(last_discovered.get('count') or 0)}; "
            f"activated {', '.join(activated_names) if activated_names else 'none'}; "
            f"resources used {len(resources)}"
        )
    subagents = [
        e for e in events if str(e.get("event", "")).startswith("subagent_")
    ]
    if subagents:
        names = sorted({str(event.get("name")) for event in subagents if event.get("name")})
        lines.append(
            "subagents: "
            f"events {len(subagents)}; "
            f"names {', '.join(names) if names else 'none'}"
        )
    lines.append(f"result: {result}")
    return "\n".join(lines)


def _summarize_context_report(report: dict[str, Any]) -> str:
    """Показываем в CLI короткую строку о последней сборке контекста."""

    history = report.get("history")
    if not isinstance(history, dict):
        history = {}
    fragments = report.get("fragments")
    request_tokens = int(report.get("request_tokens_estimate") or 0)
    max_tokens = int(report.get("max_tokens") or 0)
    tools_tokens = int(report.get("tools_tokens_estimate") or 0)
    total_entries = int(history.get("total_entries") or 0)
    rendered_entries = int(history.get("rendered_entries") or 0)
    clipped = len(history.get("clipped_tool_messages") or [])
    dropped = len(history.get("dropped_entries") or [])
    return (
        f"context: {request_tokens}/{max_tokens} estimated tokens; "
        f"tools: {tools_tokens}; "
        f"fragments: {len(fragments or [])}; "
        f"history: {rendered_entries}/{total_entries} entries; "
        f"clipped tool messages: {clipped}; dropped entries: {dropped}"
    )


def _resolve_trace_path(cfg: Config, trace_id: str) -> Path:
    """Сначала ищем точное имя trace, чтобы parent id не открывал child trace."""

    traces_dir = cfg.state_dir.joinpath("traces")
    normalized = trace_id[:-6] if trace_id.endswith(".jsonl") else trace_id
    exact = traces_dir / f"{normalized}.jsonl"
    if exact.exists():
        return exact
    matches = sorted(traces_dir.glob(f"{trace_id}*.jsonl"))
    if not matches:
        raise SystemExit(f"trace not found: {trace_id}")
    return matches[0]


def _safe_trace_label(value: str) -> str:
    """Делаем label безопасной частью имени файла trace."""

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "child"
