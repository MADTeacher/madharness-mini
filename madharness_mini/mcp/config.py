"""Чтение отдельного `.madharness-mini/mcp.json` без изменения общего Config."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..policy import Policy


@dataclass
class McpServerConfig:
    """Настройки одного явно включённого stdio MCP-сервера."""

    name: str
    command: str
    args: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: float


def load_mcp_server_configs(cfg: Config, policy: Policy) -> list[McpServerConfig]:
    """Читаем `.madharness-mini/mcp.json`; отсутствие файла означает MCP off."""

    path = cfg.state_dir / "mcp.json"
    if not path.exists():
        return []
    import json

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid MCP config JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("invalid MCP config: top-level value must be object")
    servers = raw.get("servers", {})
    if not isinstance(servers, dict):
        raise RuntimeError("invalid MCP config: servers must be object")

    configs = []
    for name, item in servers.items():
        if not isinstance(name, str) or not _safe_server_name(name):
            raise RuntimeError(f"invalid MCP server name: {name}")
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid MCP server config for {name}: expected object")
        if item.get("enabled") is not True:
            continue
        configs.append(_parse_server_config(name, item, policy))
    return configs


def _parse_server_config(
    name: str, item: dict[str, Any], policy: Policy
) -> McpServerConfig:
    """Проверяем поля одного сервера до запуска внешнего процесса."""

    command = item.get("command")
    if not isinstance(command, str) or not command.strip():
        raise RuntimeError(f"invalid MCP server {name}: command must be non-empty string")
    raw_args = item.get("args", [])
    if not isinstance(raw_args, list) or not all(
        isinstance(arg, str) for arg in raw_args
    ):
        raise RuntimeError(f"invalid MCP server {name}: args must be list of strings")
    raw_cwd = item.get("cwd", ".")
    if not isinstance(raw_cwd, str):
        raise RuntimeError(f"invalid MCP server {name}: cwd must be string")
    cwd, error = policy.safe_path(raw_cwd)
    if error:
        raise RuntimeError(f"invalid MCP server {name}: {error}")
    if cwd is None or not cwd.exists() or not cwd.is_dir():
        raise RuntimeError(f"invalid MCP server {name}: cwd is not a directory")

    raw_env = item.get("env", {})
    if not isinstance(raw_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_env.items()
    ):
        raise RuntimeError(f"invalid MCP server {name}: env must be object of strings")

    timeout = item.get("timeout_seconds", 20)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or timeout <= 0
    ):
        raise RuntimeError(
            f"invalid MCP server {name}: timeout_seconds must be positive number"
        )

    return McpServerConfig(
        name=name,
        command=command,
        args=list(raw_args),
        cwd=cwd,
        env=dict(raw_env),
        timeout_seconds=float(timeout),
    )


def _safe_server_name(name: str) -> bool:
    """Ограничиваем имя сервера символами, пригодными для tool prefix."""

    return bool(name) and all(
        char.isascii() and (char.isalnum() or char in {"_", "-"})
        for char in name
    )
