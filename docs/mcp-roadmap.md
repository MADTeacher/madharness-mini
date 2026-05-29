# Дорожная карта MCP

Этот документ описывает, как добавить в `madharness-mini` минимальную поддержку MCP без официального SDK и без внешних runtime-зависимостей. Цель - показать сам протокол и аккуратно встроить MCP tools в уже существующий `ToolRegistry`.

Официальные страницы, на которые ориентируется план:

- [MCP specification](https://modelcontextprotocol.io/specification)
- [Transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
- [Lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)
- [Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

## Цель

MCP в `madharness-mini` должен дать агенту доступ к инструментам внешнего MCP-сервера через обычный механизм tools:

```text
MCP server -> tools/list -> ToolSpec -> ToolRegistry -> model tool_call -> tools/call
```

Для модели MCP-инструмент должен выглядеть как обычный tool с JSON Schema. Для ядра харнесса это должен быть ещё один `ToolProvider`.

## Не цели V1

- Не использовать официальный MCP SDK.
- Не добавлять зависимости в `pyproject.toml`.
- Не поддерживать Streamable HTTP.
- Не поддерживать resources, prompts, sampling, elicitation, roots и progress.
- Не делать несколько transport-типов.
- Не делать auto-discovery MCP-серверов.

V1 поддерживает только stdio transport и только tools.

## Место в архитектуре

Пакет внутри основного проекта:

```text
madharness_mini/
  mcp/
    __init__.py
    config.py
    protocol.py
    stdio.py
    provider.py
    results.py
```

Роли файлов:

| Файл | Роль |
| --- | --- |
| `config.py` | Читает `.madharness-mini/mcp.json` и валидирует включённые серверы. |
| `protocol.py` | JSON-RPC 2.0 request/response, id, ошибки, валидация формы. |
| `stdio.py` | Запуск subprocess, обмен JSON-сообщениями по stdin/stdout, timeout, закрытие. |
| `provider.py` | `McpToolProvider`, который превращает MCP tools в `ToolSpec`. |
| `results.py` | Преобразует ответ `tools/call` в `ok()` или `fail()` observation. |
| `__init__.py` | Экспортирует маленький публичный API пакета. |

`run_agent()` создаёт MCP providers и передаёт их в `ToolRegistry`.

```python
mcp_providers = load_mcp_tool_providers(cfg)
registry = ToolRegistry(cfg, providers=mcp_providers)
```

## Конфиг

MCP-настройки читаются из отдельного файла `.madharness-mini/mcp.json`.
Основной `Config`, defaults и `.madharness-mini/config.json` не меняются.
Если файла `mcp.json` нет, MCP считается выключенным.

```json
{
  "servers": {
    "demo": {
      "enabled": true,
      "command": "python3",
      "args": ["scripts/demo_mcp_server.py"],
      "cwd": ".",
      "timeout_seconds": 20
    }
  }
}
```

Поля:

| Поле | Смысл |
| --- | --- |
| `enabled` | Включает сервер. По умолчанию `false`. |
| `command` | Исполняемый файл без shell-обёртки. |
| `args` | Список аргументов. |
| `cwd` | Рабочая папка сервера относительно `workspace_root`. |
| `env` | Опциональные переменные окружения, только явные ключи и значения. |
| `timeout_seconds` | Таймаут `initialize`, `tools/list` и `tools/call`. |

Почему `command` и `args` разделены: `subprocess.Popen([...], shell=False)` проще проверять и безопаснее объяснять.

## Безопасность запуска

MCP-сервер - это исполняемый процесс. Поэтому его запуск должен быть явным.

Правила V1:

- Серверы запускаются только из `.madharness-mini/mcp.json`.
- `enabled` должен быть `true`.
- `cwd` проходит через `Policy.safe_path()`.
- `command` не выполняется через shell.
- `args` не склеиваются в строку.
- Переменные окружения не наследуют секреты харнесса автоматически.
- `api_key` модели не передаётся MCP-серверу.
- stderr не отправляется модели целиком, только короткая диагностика при сбое.

Отдельный вопрос для будущего: allowlist команд MCP-серверов. В V1 можно начать с явного конфига и заметного предупреждения в документации.

## JSON-RPC subset

MCP использует JSON-RPC 2.0. В V1 нужны три типа сообщений:

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Notification:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized",
  "params": {}
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {}
}
```

Ошибочный response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

`protocol.py` должен уметь:

- выдавать монотонный `id`;
- собирать request и notification;
- отличать `result` от `error`;
- проверять совпадение `id`;
- превращать ошибку протокола в понятный `RuntimeError`.

## Lifecycle

Минимальная последовательность:

1. Запустить процесс MCP-сервера.
2. Отправить `initialize`.
3. Проверить response и capabilities.
4. Отправить `notifications/initialized`.
5. Вызвать `tools/list`.
6. Зарегистрировать tools как `ToolSpec`.
7. На каждый вызов модели отправлять `tools/call`.
8. В конце `run_agent()` закрыть процесс.

Пример `initialize`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "clientInfo": {
      "name": "madharness-mini",
      "version": "0.1.0"
    },
    "capabilities": {}
  }
}
```

Если сервер вернул другой совместимый `protocolVersion`, V1 может принять его только при точном совпадении с поддерживаемой константой. Это проще для учебного клиента.

## Stdio transport

`stdio.py` использует только стандартную библиотеку:

| Модуль | Зачем нужен |
| --- | --- |
| `subprocess` | Запускает MCP-сервер. |
| `json` | Кодирует и декодирует JSON-RPC сообщения. |
| `threading` | Отдельно читает stdout/stderr, чтобы не зависнуть. |
| `queue` | Передаёт прочитанные строки основному потоку. |
| `time` | Считает timeout. |

Правила обмена:

- Один JSON-RPC message пишется одной строкой в stdin.
- После записи нужен `flush()`.
- stdout читается построчно.
- stderr собирается отдельно и не смешивается с protocol stdout.
- Если процесс завершился до ответа, возвращается понятная ошибка.
- Если пришёл невалидный JSON, запуск MCP-сервера считается сломанным.

## `tools/list`

Ожидаемый результат содержит список tools. Для каждого tool нужны:

| MCP поле | Использование в `ToolSpec` |
| --- | --- |
| `name` | Часть имени инструмента. |
| `description` | Описание для модели. |
| `inputSchema` | `parameters` в `ToolSpec`. |

Имена нужно префиксовать, чтобы избежать коллизий:

```text
mcp__<server_name>__<tool_name>
```

Например MCP tool `search` с сервера `docs` станет tool name `mcp__docs__search`.

Оригинальное имя хранится внутри handler, чтобы `tools/call` отправлял серверу именно `search`.

## `tools/call`

Handler MCP-инструмента получает обычные args от модели и отправляет:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "context budget"
    }
  }
}
```

Результат MCP нужно привести к observation харнесса:

```json
{
  "ok": true,
  "tool": "mcp__docs__search",
  "summary": "MCP tool docs.search completed",
  "content": "..."
}
```

Если MCP вернул `isError: true`, observation должен быть `fail()`, даже если JSON-RPC response технически успешен.

## Преобразование результатов

V1 поддерживает текстовые результаты:

| MCP content type | Что делать |
| --- | --- |
| `text` | Добавить в `content`. |
| `image` | Вернуть metadata: mime type и размер, без base64. |
| `resource` | Вернуть короткое описание, без загрузки ресурса. |
| неизвестный type | Добавить диагностическую строку. |

`structuredContent`, если он есть, можно положить в поле `data`, предварительно убедившись, что это JSON-совместимое значение.

Все большие строки проходят через `clipped()`, чтобы MCP-сервер не переполнил контекст.

## Закрытие процессов

Для MCP выбран явный lifecycle у `ToolRegistry`: метод `close()` вызывает
`close()` у providers, если он у них есть. `run_agent()` закрывает registry через
`finally`, поэтому stdio-процесс завершается и при финальном ответе, и при ошибке.

```python
registry = ToolRegistry(cfg, providers=mcp_providers)
try:
    ...
finally:
    registry.close()
```

`McpToolProvider.close()` должен:

1. Закрыть stdin.
2. Подождать короткий timeout.
3. Если процесс жив, вызвать `terminate()`.
4. Если всё ещё жив, вызвать `kill()`.

## Трассы

Новые события:

| Event | Данные |
| --- | --- |
| `mcp_server_started` | `server`, `command`, `tools_count` после list. |
| `mcp_server_error` | `server`, короткая причина. |
| `mcp_server_stopped` | `server`, exit code. |

В `tool_observation` уже попадёт результат MCP tool через общий путь `ToolRegistry.call()`.

Секреты, env и большие stderr/stdout в трассу писать нельзя.

## Тест-план

Новые тесты:

| Тест | Что проверяет |
| --- | --- |
| `test_jsonrpc_request_has_incrementing_id` | `protocol.py` выдаёт корректные JSON-RPC requests. |
| `test_jsonrpc_error_raises_runtime_error` | Error response превращается в понятную ошибку. |
| `test_stdio_client_initialize_and_list_tools` | Fake MCP server отвечает на lifecycle и `tools/list`. |
| `test_mcp_provider_exports_toolspecs` | MCP tools превращаются в префиксованные `ToolSpec`. |
| `test_mcp_tool_call_returns_ok_observation` | `tools/call` становится `ok()` observation. |
| `test_mcp_tool_call_iserror_returns_fail` | `isError: true` становится `fail()`. |
| `test_mcp_process_is_closed_after_run` | Процесс закрывается в `finally`. |
| `test_mcp_config_rejects_unsafe_cwd` | `cwd` вне workspace запрещён. |

Fake server можно сделать маленьким Python-скриптом в тестах. Он читает строки JSON из stdin и пишет JSON responses в stdout.

## Этапы внедрения

1. Добавить `madharness_mini/mcp/protocol.py` и unit-тесты без subprocess.
2. Добавить `stdio.py` с fake server тестом.
3. Добавить parser для `.madharness-mini/mcp.json`.
4. Добавить `McpToolProvider`.
5. Добавить result conversion в `results.py`.
6. Добавить `ToolRegistry.close()`.
7. Обернуть `run_agent()` в `try/finally` для закрытия providers.
8. Добавить трассировку старта, ошибок и остановки MCP-серверов.
9. Обновить `docs/capabilities.md` и `docs/code-overview.md`.

## Готовность

V1 считается готовой, когда:

- Один stdio MCP-сервер может дать tools через `tools/list`.
- Модель может вызвать MCP tool через текущий `ToolRegistry`.
- Процесс MCP-сервера закрывается после `run`.
- MCP не добавляет зависимостей в `pyproject.toml`.
- Старые проекты без `.madharness-mini/mcp.json` работают как раньше.
- Ошибки MCP видны в trace и не ломают формат observations.
