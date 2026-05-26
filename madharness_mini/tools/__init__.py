"""Пакет инструментов агента: диспетчер, спецификации, поставщики"""

from .registry import ToolRegistry
from .specs import ToolProvider, ToolSpec

__all__ = ["ToolProvider", "ToolRegistry", "ToolSpec"]
