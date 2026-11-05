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
    """Запускает включенные MCP-серверы и отдает их tools как ToolSpec."""

    def __init__(self):
        # Список клиентов MCP-серверов
        self.clients: list[StdioMcpClient] = [] 

    def specs(self, ctx: ToolContext) -> Iterable[ToolSpec]:
        """Читаем `.madharness-mini/mcp.json` и регистрируем доступные MCP tools."""

        specs: list[ToolSpec] = []
        # Загружаем конфигурации MCP-серверов
        # и создаем клиентов для каждого из них
        for config in load_mcp_server_configs(ctx.cfg, ctx.policy):
            client = StdioMcpClient(config) # Создаем клиента для сервера
            try:
                # Запускаем сервер и получаем его tools
                tools = client.start()
            except Exception as exc:
                # Если возникла ошибка, записываем ее в трассировку
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
            
            # Добавляем клиента в список
            self.clients.append(client)
            if ctx.trace:
                ctx.trace.write(
                    "mcp_server_started",
                    server=config.name,
                    command=config.command,
                    tools_count=len(tools),
                )

            # Преобразуем tools/list одного сервера в локальные ToolSpec
            specs.extend(_tool_specs_for_server(config, client, tools))
        # Возвращаем список ToolSpec для всех серверов
        return specs

    def close(self, trace: Any | None = None) -> None:
        """Закрываем все MCP subprocess после run, 
        включая ошибочные сценарии."""

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
    for tool in tools: # перебираем все tools одного сервера
        # если tool не является словарем, выбрасываем ошибку
        if not isinstance(tool, dict): 
            raise RuntimeError(
                f"invalid MCP tool from {config.name}: expected object",
            )
        # Получаем имя инструмента
        original_name = tool.get("name")

        # если имя не является строкой или оно пустое, 
        # то выбрасываем ошибку
        if not isinstance(original_name, str) or not original_name:
            raise RuntimeError(
                f"invalid MCP tool from {config.name}: missing name",
            )

        # Преобразуем имя инструмента в имя для модели
        exported_name = exported_tool_name(config.name, original_name)

        # если имя для модели уже было использовано, выбрасываем ошибку
        if exported_name in seen:
            raise RuntimeError(
                f"duplicate exported MCP tool name: {exported_name}",
            )
        seen.add(exported_name) # Добавляем имя для модели в множество

        # Получаем описание инструмента и его параметры
        description = tool.get("description")
        parameters = tool.get("inputSchema") or {
            "type": "object",
            "properties": {},
        }
        if not isinstance(description, str):
            description = f"MCP tool {config.name}.{original_name}"
        if not isinstance(parameters, dict):
            raise RuntimeError(
                f"invalid MCP tool {config.name}.{original_name}: "
                f"inputSchema must be object"
            )

        # Формируем ToolSpec для инструмента и добавляем его в список
        specs.append(
            ToolSpec(
                name=exported_name,
                description=f"[MCP:{config.name}] {description}",
                parameters=parameters,
                handler=_handler(
                    client,
                    config.name,
                    original_name,
                    exported_name,
                ),
            )
        )

    # Возвращаем список ToolSpec для всех инструментов
    return specs


def _handler(
    client: StdioMcpClient,
    server_name: str,
    tool_name: str,
    exported_name: str,
):
    """Создаем handler с исходным MCP tool name, скрытым от модели."""

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
    """Собираем имя, под которым модель увидит инструмент, 
    предоставляемый MCP-сервером, в списке функций.

    Chat Completions принимает function name только из латиницы, цифр,
    ``_`` и ``-``, не длиннее ``TOOL_NAME_MAX_LENGTH`` символов. 
    Имя с сервера приходит как есть, поэтому недопустимые символы заменяем на
    ``_`` и добавляем префикс ``mcp__<сервер>__``: так инструмент не столкнется
    с встроенным инструментом и с одноименным инструментом другого сервера.
    Имя сервера к этому моменту уже проверено в конфиге и в префикс
    попадает без замены. Handler вызывает сервер исходным именем.
    Пустая строка после замены и имя длиннее лимита останавливают регистрацию:
    такой инструмент в реестр не попадает.
    """

    safe_tool = re.sub(r"[^A-Za-z0-9_-]", "_", tool_name)
    if not safe_tool:
        raise RuntimeError(f"invalid MCP tool name: {tool_name}")
    exported = f"mcp__{server_name}__{safe_tool}"
    if len(exported) > TOOL_NAME_MAX_LENGTH:
        raise RuntimeError(
            f"MCP tool name is too long for model API: {exported}",
        )
    return exported
