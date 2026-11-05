"""Выставляем наружу пакета публичный API минимального MCP-клиента для подключения по stdio"""


from .provider import McpToolProvider

__all__ = ["McpToolProvider"]
