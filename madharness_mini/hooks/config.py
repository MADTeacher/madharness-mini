"""Чтение `.madharness-mini/hooks.json` с пользовательскими command hooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..policy import Policy
from .types import HOOK_EVENTS


@dataclass(frozen=True)
class CommandHookConfig:
    """Один пользовательский hook, запускаемый как локальная команда."""

    id: str
    event: str
    match: dict[str, Any]
    command: str
    args: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: float


def load_command_hook_configs(cfg: Config, policy: Policy) -> list[CommandHookConfig]:
    """Читаем hooks.json; отсутствие файла означает, что hooks выключены."""

    path = cfg.state_dir / "hooks.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid hooks config JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("invalid hooks config: top-level value must be object")
    hooks = raw.get("hooks", [])
    if not isinstance(hooks, list):
        raise RuntimeError("invalid hooks config: hooks must be list")
    configs = []
    for index, item in enumerate(hooks, 1):
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid hook #{index}: expected object")
        if item.get("enabled", True) is not True:
            continue
        configs.append(_parse_hook_config(index, item, policy))
    return configs


def _parse_hook_config(
    index: int,
    item: dict[str, Any],
    policy: Policy,
) -> CommandHookConfig:
    """Проверяем один hook до запуска внешней команды."""

    hook_id = item.get("id")
    if not isinstance(hook_id, str) or not _safe_hook_id(hook_id):
        raise RuntimeError(f"invalid hook #{index}: id must be non-empty safe string")
    event = item.get("event")
    if not isinstance(event, str) or event not in HOOK_EVENTS:
        allowed = ", ".join(sorted(HOOK_EVENTS))
        raise RuntimeError(f"invalid hook {hook_id}: event must be one of: {allowed}")
    command = item.get("command")
    if not isinstance(command, str) or not command.strip():
        raise RuntimeError(f"invalid hook {hook_id}: command must be non-empty string")
    raw_args = item.get("args", [])
    if not isinstance(raw_args, list) or not all(
        isinstance(arg, str) for arg in raw_args
    ):
        raise RuntimeError(f"invalid hook {hook_id}: args must be list of strings")
    raw_cwd = item.get("cwd", ".")
    if not isinstance(raw_cwd, str):
        raise RuntimeError(f"invalid hook {hook_id}: cwd must be string")
    cwd, error = policy.safe_path(raw_cwd)
    if error:
        raise RuntimeError(f"invalid hook {hook_id}: {error}")
    if cwd is None or not cwd.exists() or not cwd.is_dir():
        raise RuntimeError(f"invalid hook {hook_id}: cwd is not a directory")

    raw_env = item.get("env", {})
    if not isinstance(raw_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_env.items()
    ):
        raise RuntimeError(f"invalid hook {hook_id}: env must be object of strings")

    raw_match = item.get("match", {})
    if not isinstance(raw_match, dict) or not all(
        isinstance(key, str) and _valid_match_value(value)
        for key, value in raw_match.items()
    ):
        raise RuntimeError(
            f"invalid hook {hook_id}: match must be object with scalar values"
        )

    timeout = item.get("timeout_seconds", 5)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or timeout <= 0
    ):
        raise RuntimeError(
            f"invalid hook {hook_id}: timeout_seconds must be positive number"
        )

    return CommandHookConfig(
        id=hook_id,
        event=event,
        match=dict(raw_match),
        command=command,
        args=list(raw_args),
        cwd=cwd,
        env=dict(raw_env),
        timeout_seconds=float(timeout),
    )


def _safe_hook_id(value: str) -> bool:
    """Делаем id пригодным для trace и сообщений об ошибках."""

    return bool(value.strip()) and all(
        char.isascii() and (char.isalnum() or char in {"_", "-", "."})
        for char in value
    )


def _valid_match_value(value: Any) -> bool:
    """Разрешаем простые exact-match значения без вложенного языка правил."""

    scalar = str | int | float | bool
    if value is None or isinstance(value, scalar):
        return True
    if isinstance(value, list):
        return all(item is None or isinstance(item, scalar) for item in value)
    return False
