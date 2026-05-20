"""Диспетчер инструментов: регистрация, схемы и безопасный вызов."""

from typing import Any

from ..config import Config
from ..policy import Policy
from ..utils import fail
from .builtins import builtin_specs
from .context import ToolContext


class ToolRegistry:
    """Все инструменты run: вызов по имени и единая обработка сбоев.

    Реальные handlers живут в отдельных модулях пакета. Registry только хранит
    ToolSpec, отдаёт схемы модели и превращает исключения в fail-наблюдения.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.policy = Policy(cfg)
        self.context = ToolContext(cfg, self.policy)
        self.tools = {tool.name: tool for tool in builtin_specs()}

    def schemas(self) -> list[dict[str, Any]]:
        """Список схем для поля tools в запросе к модели."""

        return [tool.schema() for tool in self.tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Вызываем handler по имени; сбои ловим и отдаём fail, не падаем."""

        tool = self.tools.get(name)
        if not tool:
            return fail(name, "unknown tool")
        try:
            return tool.handler(self.context, args)
        except Exception as exc:
            return fail(name, f"{type(exc).__name__}: {exc}")
