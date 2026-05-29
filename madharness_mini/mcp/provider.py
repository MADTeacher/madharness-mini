"""ToolProvider, который адаптирует MCP tools к ToolRegistry."""

import re
from collections.abc import Iterable
from typing import Any

from ..tools.context import ToolContext
from ..tools.specs import ToolSpec
from ..utils import clipped
from .config import McpServerConfig, load_mcp_server_configs
from .results import mcp_result_to_observation
from .stdio import StdioMcpClient

TOOL_NAME_MAX_LENGTH = 64


class McpToolProvider:
    """Запускает включённые MCP-серверы и отдаёт их tools как ToolSpec."""

    def __init__(self):
        self.clients: list[StdioMcpClient] = []

    def specs(self, ctx: ToolContext) -> Iterable[ToolSpec]:
        """Читаем `.madharness-mini/mcp.json` и регистрируем доступные MCP tools."""

        specs: list[ToolSpec] = []
        for config in load_mcp_server_configs(ctx.cfg, ctx.policy):
            client = StdioMcpClient(config)
            try:
                tools = client.start()
            except Exception as exc:
                if ctx.trace:
                    ctx.trace.write(
                        "mcp_server_error",
                        server=config.name,
                        error=clipped(str(exc), 1000),
                    )
                client.close()
                raise RuntimeError(
                    f"MCP server {config.name} failed to start: {exc}"
                ) from exc
            self.clients.append(client)
            if ctx.trace:
                ctx.trace.write(
                    "mcp_server_started",
                    server=config.name,
                    command=config.command,
                    tools_count=len(tools),
                )
            specs.extend(_tool_specs_for_server(config, client, tools))
        return specs

    def close(self, trace: Any | None = None) -> None:
        """Закрываем все MCP subprocess после run, включая ошибочные сценарии."""

        for client in self.clients:
            code = client.close()
            if trace:
                trace.write(
                    "mcp_server_stopped",
                    server=client.config.name,
                    exit_code=code,
                )


def _tool_specs_for_server(
    config: McpServerConfig,
    client: StdioMcpClient,
    tools: list[dict[str, Any]],
) -> list[ToolSpec]:
    """Преобразуем tools/list одного сервера в локальные ToolSpec."""

    specs = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise RuntimeError(f"invalid MCP tool from {config.name}: expected object")
        original_name = tool.get("name")
        if not isinstance(original_name, str) or not original_name:
            raise RuntimeError(f"invalid MCP tool from {config.name}: missing name")
        exported_name = exported_tool_name(config.name, original_name)
        if exported_name in seen:
            raise RuntimeError(f"duplicate exported MCP tool name: {exported_name}")
        seen.add(exported_name)
        description = tool.get("description")
        parameters = tool.get("inputSchema") or {"type": "object", "properties": {}}
        if not isinstance(description, str):
            description = f"MCP tool {config.name}.{original_name}"
        if not isinstance(parameters, dict):
            raise RuntimeError(
                f"invalid MCP tool {config.name}.{original_name}: inputSchema must be object"
            )
        specs.append(
            ToolSpec(
                name=exported_name,
                description=f"[MCP:{config.name}] {description}",
                parameters=parameters,
                handler=_handler(client, config.name, original_name, exported_name),
            )
        )
    return specs


def _handler(
    client: StdioMcpClient,
    server_name: str,
    tool_name: str,
    exported_name: str,
):
    """Создаём handler с исходным MCP tool name, скрытым от модели."""

    def call(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        result = client.call_tool(tool_name, args)
        return mcp_result_to_observation(
            exported_name,
            server_name,
            tool_name,
            result,
        )

    return call


def exported_tool_name(server_name: str, tool_name: str) -> str:
    """Делаем имя MCP tool совместимым с OpenAI function name."""

    safe_tool = re.sub(r"[^A-Za-z0-9_-]", "_", tool_name)
    if not safe_tool:
        raise RuntimeError(f"invalid MCP tool name: {tool_name}")
    exported = f"mcp__{server_name}__{safe_tool}"
    if len(exported) > TOOL_NAME_MAX_LENGTH:
        raise RuntimeError(f"MCP tool name is too long for model API: {exported}")
    return exported
