"""Общий контекст, который получают handlers инструментов."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..policy import Policy


@dataclass
class ToolContext:
    """Держим настройки и политику безопасности для одного запуска run.

    Registry создаёт этот объект один раз, а handlers используют его для доступа
    к workspace, защищённым путям и правилам shell-команд.
    """

    cfg: Config
    policy: Policy
    trace: Any | None = None
    skill_runtime: Any | None = None
    writable_suffixes: tuple[str, ...] | None = None
    write_scope_description: str = ""

    def write_path_error(self, raw: str) -> str:
        """Проверяем дополнительный scope записи для специализированных запусков."""

        if self.writable_suffixes is None:
            return ""
        lowered = raw.lower()
        if any(lowered.endswith(suffix) for suffix in self.writable_suffixes):
            return ""
        description = self.write_scope_description or (
            "this agent may write only files with allowed suffixes: "
            + ", ".join(self.writable_suffixes)
        )
        return f"write path denied by scope: {description}: {raw}"


def normalize_writable_suffixes(values: Iterable[str] | None) -> tuple[str, ...] | None:
    """Нормализуем список разрешённых суффиксов записи для ToolContext."""

    if values is None:
        return None
    suffixes = tuple(str(value).lower() for value in values if str(value).strip())
    return suffixes or None
