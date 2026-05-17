"""Общие маленькие помощники для инструментов, схем и ответов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_DIR = ".madharness-mini"
MAX_OUTPUT = 12000

DEFAULT_CONFIG = {
    "model": "deepseek/deepseek-v4-flash",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "",
    "temperature": 0.2,
    "max_turns": 8,
    "workspace_root": ".",
    "protected_paths": [".git", ".env", "secrets", "~/.ssh"],
    "allow_shell": True,
}


def clipped(text: str, limit: int = MAX_OUTPUT) -> str:
    """Обрезать длинный текстовый результат инструмента до безопасного размера."""

    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped {len(text) - limit} chars]"


def ok(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    """Сформировать успешное наблюдение инструмента для модели."""

    return {"ok": True, "tool": tool, "summary": summary, **data}


def fail(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    """Сформировать ошибочное наблюдение инструмента для модели."""

    return {"ok": False, "tool": tool, "summary": summary, **data}


def ignored(path: Path) -> bool:
    """Проверить, надо ли скрыть путь от поиска и листинга файлов."""

    ignored_names = {".git", STATE_DIR, "__pycache__", ".venv", ".uv-cache"}
    return any(part in ignored_names for part in path.parts)


def parse_tool_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Достать имя инструмента и JSON-аргументы из tool call модели."""

    fn = call.get("function", {})
    raw = fn.get("arguments") or "{}"
    args = raw if isinstance(raw, dict) else json.loads(raw)
    return fn.get("name", ""), args


def obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Создать JSON Schema для объекта аргументов инструмента."""

    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


def strp(
    default: str | None = None, desc: str = "", req: bool = False
) -> dict[str, Any]:
    """Создать JSON Schema для строкового параметра инструмента."""

    data: dict[str, Any] = {"type": "string", "description": desc}
    if default is not None and not req:
        data["default"] = default
    return data


def intp(default: int) -> dict[str, Any]:
    """Создать JSON Schema для целочисленного параметра с default."""

    return {"type": "integer", "default": default}
