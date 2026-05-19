"""JSONL-трассы запусков ask/run и команда trace для просмотра."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import Config


class Trace:
    """Один файл трассы на сессию: события по мере работы ask или run."""

    def __init__(self, cfg: Config, kind: str):
        cfg.ensure_dirs()
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.path = cfg.state_dir / "traces" / f"{self.id}.jsonl"
        self.write("session_start", kind=kind)

    def write(self, event: str, **data: Any) -> None:
        """Дописываем одно событие в конец JSONL (модель, tool, конец сессии)."""

        record = {"ts": time.time(), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_trace(cfg: Config, trace_id: str) -> str:
    """Текстовая сводка для CLI: путь, число событий, tool calls, итог.

    trace_id может быть префиксом имени файла в каталоге traces/.
    """

    matches = list((cfg.state_dir / "traces").glob(f"{trace_id}*.jsonl"))
    if not matches:
        raise SystemExit(f"trace not found: {trace_id}")
    path = matches[0]
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tools = [e for e in events if e.get("event") == "tool_observation"]
    end = next((e for e in reversed(events) if e.get("event") == "session_end"), {})
    result = str(end.get("result", ""))[:1000]
    return f"trace: {path}\nevents: {len(events)}\ntool calls: {len(tools)}\nresult: {result}"
