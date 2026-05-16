from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_DIR = ".madharness-mini"
MAX_OUTPUT = 12000

DEFAULT_CONFIG = {
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4-flash",
    "base_url": "",
    "api_key": "",
    "providers": {
        "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": ""},
        "kodikrouter": {"base_url": "https://kodikrouter.ru/api/v1", "api_key": ""},
        "local": {"base_url": "http://127.0.0.1:11434/v1", "api_key": ""},
    },
    "temperature": 0.2,
    "max_turns": 8,
    "workspace_root": ".",
    "protected_paths": [".git", ".env", "secrets", "~/.ssh"],
    "allow_shell": True,
    "verify_command": "uv run -m unittest discover -s tests",
}


def clipped(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped {len(text) - limit} chars]"


def ok(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    return {"ok": True, "tool": tool, "summary": summary, **data}


def fail(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    return {"ok": False, "tool": tool, "summary": summary, **data}


def ignored(path: Path) -> bool:
    ignored_names = {".git", STATE_DIR, "__pycache__", ".venv", ".uv-cache"}
    return any(part in ignored_names for part in path.parts)


def parse_tool_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function", {})
    raw = fn.get("arguments") or "{}"
    args = raw if isinstance(raw, dict) else json.loads(raw)
    return fn.get("name", ""), args


def obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


def strp(
    default: str | None = None, desc: str = "", req: bool = False
) -> dict[str, Any]:
    data: dict[str, Any] = {"type": "string", "description": desc}
    if default is not None and not req:
        data["default"] = default
    return data


def intp(default: int) -> dict[str, Any]:
    return {"type": "integer", "default": default}
