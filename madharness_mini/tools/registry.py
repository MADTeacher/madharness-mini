"""Диспетчер инструментов: регистрация, схемы и безопасный вызов."""

from collections.abc import Iterable
from typing import Any

from ..config import Config
from ..policy import Policy
from ..utils import fail
from .builtins import BuiltinToolProvider
from .context import ToolContext, normalize_writable_suffixes
from .specs import ToolProvider


class ToolRegistry:
    """Все инструменты run: вызов по имени и единая обработка сбоев.

    Реальные handlers живут в отдельных модулях пакета. Registry только хранит
    ToolSpec, отдаёт схемы модели и превращает исключения в fail-наблюдения.
    """

    def __init__(
        self,
        cfg: Config,
        # используем для регистрации внешних инструментов, не
        # входящих в стандартный набор встроенных инструментов
        providers: Iterable[ToolProvider] | None = None,
        trace: Any | None = None,
        skill_runtime: Any | None = None,
        allowed_tools: Iterable[str] | None = None,
        writable_suffixes: Iterable[str] | None = None,
        write_scope_description: str = "",
    ):
        self.cfg = cfg
        self.policy = Policy(cfg)
        self.context = ToolContext(
            cfg,
            self.policy,
            trace,
            skill_runtime,
            normalize_writable_suffixes(writable_suffixes),
            write_scope_description,
        )
        self.providers = [BuiltinToolProvider(), *list(providers or [])]
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.tools = {}
        try:
            for provider in self.providers:
                # получаем список ToolSpec от каждого provider
                # и регистрируем их в словаре self.tools
                for tool in provider.specs(self.context):
                    if self.allowed_tools is not None and tool.name not in self.allowed_tools:
                        continue
                    if tool.name in self.tools:
                        raise RuntimeError(f"duplicate tool name: {tool.name}")
                    self.tools[tool.name] = tool
        except Exception:
            self.close()
            raise

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

    def close(self) -> None:
        """Закрываем providers с lifecycle, например MCP subprocess."""

        for provider in self.providers:
            close = getattr(provider, "close", None)
            if close is None:
                continue
            try:
                close(self.context.trace)
            except TypeError:
                close()
            except Exception as exc:
                if self.context.trace:
                    self.context.trace.write(
                        "tool_provider_close_error",
                        provider=type(provider).__name__,
                        error=str(exc),
                    )
