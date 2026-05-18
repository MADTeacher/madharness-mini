"""Общие константы и функции для схем, наблюдений и ограничений вывода."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_DIR = ".madharness-mini"
MAX_OUTPUT = 20000

DEFAULT_CONFIG = {
    "model": "deepseek/deepseek-v4-flash",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "",
    "temperature": 0.2,
    "max_turns": 50,
    "workspace_root": ".",
    "protected_paths": [".git", ".env", "secrets", "~/.ssh"],
    "allow_shell": True,
}


def clipped(text: str, limit: int = MAX_OUTPUT) -> str:
    """Ограничить текстовый результат инструмента максимальной длиной."""

    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped {len(text) - limit} chars]"


def ok(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    """Сформировать успешное JSON-наблюдение инструмента."""

    return {"ok": True, "tool": tool, "summary": summary, **data}


def fail(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    """Сформировать JSON-наблюдение об ошибке инструмента."""

    return {"ok": False, "tool": tool, "summary": summary, **data}


def ignored(path: Path) -> bool:
    """Определить, нужно ли исключить путь из поиска и листинга."""

    ignored_names = {".git", STATE_DIR, "__pycache__", ".venv", ".uv-cache"}
    return any(part in ignored_names for part in path.parts)


def parse_tool_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Извлечь имя инструмента и аргументы из `tool_call` модели."""

    fn = call.get("function", {})
    raw = fn.get("arguments") or "{}"
    args = raw if isinstance(raw, dict) else json.loads(raw)
    return fn.get("name", ""), args


def obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Создать JSON Schema объекта для аргументов инструмента."""

    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


def strp(
    default: str | None = None, desc: str = "", req: bool = False
) -> dict[str, Any]:
    """Создать JSON Schema строкового параметра инструмента."""

    data: dict[str, Any] = {"type": "string", "description": desc}
    if default is not None and not req:
        data["default"] = default
    return data


def intp(default: int) -> dict[str, Any]:
    """Создать JSON Schema целочисленного параметра со значением по умолчанию."""

    return {"type": "integer", "default": default}
