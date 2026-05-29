"""Общий контекст, который получают handlers инструментов."""

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
