# MCP в madharness-mini

`madharness-mini` умеет подключать внешние stdio MCP-серверы в режиме `run`.
Для модели такие серверы выглядят как обычные tools харнесса: они попадают в
общий список схем, вызываются через `tool_calls` и возвращают привычные
observations с полями `ok`, `tool`, `summary` и дополнительными данными.

Реализация намеренно минимальная: только stdio transport, только MCP tools и
только стандартная библиотека Python. Основной `config.json` не меняется; MCP
включается отдельным файлом `.madharness-mini/mcp.json`.

## Быстрый пример

Создайте в проекте файл:

```text
.madharness-mini/mcp.json
```

Пример с тремя reference MCP-серверами:

```json
{
  "servers": {
    "time": {
      "enabled": true,
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=Europe/Moscow"],
      "cwd": ".",
      "timeout_seconds": 90
    },
    "memory": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "cwd": ".",
      "timeout_seconds": 90
    },
    "fetch": {
      "enabled": true,
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "cwd": ".",
      "timeout_seconds": 90
    }
  }
}
```

После этого можно запустить:

```bash
madharness-mini run "Через MCP узнай текущее время, запиши факт в memory и скачай https://example.com"
```

Если файл `mcp.json` отсутствует, MCP полностью выключен и старые проекты
работают с прежним набором инструментов.

## Формат mcp.json

Корневое поле `servers` содержит объект, где ключ — локальное имя MCP-сервера.
Имя должно состоять из ASCII-букв, цифр, `_` или `-`.

| Поле | Обязательность | Смысл |
| --- | --- | --- |
| `enabled` | Да | Сервер запускается только при точном значении `true`. |
| `command` | Да | Исполняемая команда, например `npx`, `uvx`, `python`. |
| `args` | Нет | Список строковых аргументов команды. По умолчанию пустой список. |
| `cwd` | Нет | Рабочий каталог сервера внутри `workspace_root`. По умолчанию `"."`. |
| `env` | Нет | Явные переменные окружения для сервера. Значения должны быть строками. |
| `timeout_seconds` | Нет | Таймаут одного MCP-запроса. По умолчанию `20`. |

Команда и аргументы не склеиваются в shell-строку. Харнесс запускает процесс как
`[command, *args]`, поэтому shell-операторы, пайпы и подстановки тут не работают.
Если серверу нужны сложные действия, лучше вынести их в отдельный скрипт внутри
workspace и указать его как аргумент.

Пример для Playwright MCP:

```json
{
  "servers": {
    "playwright": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--browser=chrome", "--headless"],
      "cwd": ".",
      "timeout_seconds": 90
    }
  }
}
```

Playwright требует установленный браузер и часто нуждается в предварительном
прогреве окружения. Для простой демонстрации MCP лучше начинать с `time`,
`memory` и `fetch`.

## Как проходит запуск

В `run_agent()` всегда добавляется `McpToolProvider`. Если `mcp.json` нет, он
ничего не регистрирует. Если файл есть, поток такой:

1. `madharness_mini.mcp.config` читает `.madharness-mini/mcp.json`.
2. Для каждого `enabled: true` сервера проверяются `command`, `args`, `cwd`,
   `env` и `timeout_seconds`.
3. `cwd` проходит через общую `Policy.safe_path()`, поэтому сервер не может быть
   запущен из каталога вне workspace.
4. `StdioMcpClient` запускает subprocess без shell.
5. Для stdout и stderr создаются отдельные reader-потоки.
6. Клиент отправляет `initialize` с протоколом `2025-11-25`.
7. Сервер должен вернуть тот же `protocolVersion`; иначе запуск считается
   ошибочным.
8. Клиент отправляет `notifications/initialized`.
9. Клиент вызывает `tools/list`.
10. Каждый MCP tool превращается в локальный `ToolSpec`.
11. `ToolRegistry` добавляет эти specs рядом со встроенными tools.
12. Модель видит MCP tools в обычном поле `tools` OpenAI-совместимого запроса.

Если сервер не стартовал, вернул невалидный JSON, не ответил до таймаута или
прислал некорректный `tools/list`, запуск `run` завершается понятной ошибкой.
При ошибке регистрации уже запущенные providers закрываются через
`ToolRegistry.close()`.

## Имена инструментов

MCP tool получает имя:

```text
mcp__<server_name>__<tool_name>
```

Например:

| MCP-сервер | Исходный tool | Имя для модели |
| --- | --- | --- |
| `time` | `get_current_time` | `mcp__time__get_current_time` |
| `memory` | `read_graph` | `mcp__memory__read_graph` |
| `fetch` | `fetch` | `mcp__fetch__fetch` |
| `playwright` | `browser_navigate` | `mcp__playwright__browser_navigate` |

Символы в имени tool, несовместимые с OpenAI function name, заменяются на `_`.
Итоговое имя ограничено 64 символами. Оригинальное MCP-имя сохраняется внутри
handler и используется при настоящем `tools/call`.

Описание tool получает префикс `[MCP:<server>]`, а параметры берутся из
`inputSchema`. Если `inputSchema` отсутствует, используется пустая объектная
схема.

## Вызов MCP tool

Когда модель вызывает `mcp__server__tool`, `ToolRegistry.call()` идёт обычным
путём:

1. Находит `ToolSpec` по экспортированному имени.
2. Handler отправляет MCP-запрос `tools/call`.
3. В `params.name` передаётся исходное имя MCP tool.
4. В `params.arguments` передаются аргументы модели без дополнительной
   трансформации.
5. Ответ MCP преобразуется в observation харнесса.

Пример MCP-запроса:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "fetch",
    "arguments": {
      "url": "https://example.com",
      "max_length": 1000
    }
  }
}
```

## Как MCP-ответ становится observation

`madharness_mini.mcp.results` приводит `tools/call` result к формату, который уже
понимает агентский цикл.

| MCP content | Что получает модель |
| --- | --- |
| `text` | Строки объединяются в `content` и обрезаются общим лимитом `clipped()`. |
| `structuredContent` | Попадает в поле `data`. |
| `image` и `audio` | Передаются только метаданные: тип, MIME type и размер base64-строки. |
| `resource` и `resource_link` | Передаются короткие метаданные ресурса: `uri`, `name`, MIME type, description. |
| Неизвестный content type | Попадает в `diagnostics`. |

Если MCP-сервер вернул `isError: true`, observation будет ошибочным
`ok: false`, даже если JSON-RPC ответ технически успешен. Если ломается сам
transport или JSON-RPC, `ToolRegistry.call()` ловит исключение и возвращает
обычный `fail()` observation.

## Окружение и безопасность

MCP-сервер — это внешний процесс, поэтому его запуск должен быть явным и
локальным для проекта.

Харнесс наследует только небольшой безопасный набор системных переменных:
`PATH`, `HOME`, `USER`, `TMPDIR`, `TEMP`, `TMP`, `LANG`, `LC_ALL`, `ComSpec`,
`SystemRoot`, `WINDIR`. Затем добавляется явный `env` из `mcp.json`.

Переменные `MADHARNESS_MINI_*` и ключ LLM API не передаются MCP-серверам
автоматически. Если конкретному серверу нужен токен, его нужно указать явно в
`env`, понимая, что это доверенный локальный процесс.

`cwd` обязан быть существующей директорией внутри `workspace_root`. Это не
запрещает самому MCP-серверу делать сетевые запросы или читать файлы, если его
собственная логика это позволяет; поэтому подключайте только те серверы, которым
вы доверяете.

## Трассы

MCP пишет отдельные события в JSONL-трассу:

| Событие | Когда пишется | Поля |
| --- | --- | --- |
| `mcp_server_started` | Сервер прошёл `initialize` и `tools/list`. | `server`, `command`, `tools_count` |
| `mcp_server_error` | Сервер не смог стартовать или отдать tools. | `server`, `error` |
| `mcp_server_stopped` | Provider закрывает subprocess. | `server`, `exit_code` |
| `tool_observation` | Модель вызвала MCP tool. | `tool`, `args`, `observation` |

Полный stdout/stderr MCP-сервера в трассу не пишется. Для ошибок stderr
добавляется только коротким фрагментом в текст исключения.

## Закрытие процессов

`ToolRegistry.close()` вызывается в `run_agent()` через `finally`. MCP provider
закрывает каждый subprocess так:

1. Закрывает stdin сервера.
2. Ждёт штатного завершения.
3. Если сервер завис, вызывает `terminate()`.
4. Если процесс всё ещё жив, вызывает `kill()`.
5. Закрывает pipes и коротко дожидается reader-потоков.

Поэтому MCP-процессы должны закрываться и при успешном финальном ответе, и при
ошибке внутри агентского цикла.

## Ограничения текущей версии

Поддерживаются:

- stdio transport;
- `initialize`;
- `notifications/initialized`;
- `tools/list`;
- `tools/call`;
- базовая обработка server-to-client requests через ответ `Method not found`.

Не поддерживаются:

- Streamable HTTP и SSE transport;
- MCP resources как отдельная возможность клиента;
- MCP prompts;
- sampling;
- roots;
- elicitation;
- progress notifications;
- tasks.

Некоторые серверы могут писать служебные сообщения в stdout. Для stdio MCP stdout
является protocol channel, поэтому любой не-JSON текст ломает текущий запрос.
На практике это чаще всего происходит при первом запуске серверов, которые
докачивают зависимости. Для демонстраций полезно заранее прогреть такие команды
или выбрать серверы, которые сразу говорят чистым JSON-RPC.

## Быстрая проверка

После настройки `mcp.json` можно проверить, какие MCP tools увидит registry:

```bash
python -c "from madharness_mini.config import Config; from madharness_mini.mcp import McpToolProvider; from madharness_mini.tools import ToolRegistry; r=ToolRegistry(Config(), providers=[McpToolProvider()]); print('\n'.join(s['function']['name'] for s in r.schemas() if s['function']['name'].startswith('mcp__'))); r.close()"
```

Ожидаемый результат для примера `time`, `memory`, `fetch`:

```text
mcp__time__get_current_time
mcp__time__convert_time
mcp__memory__create_entities
mcp__memory__create_relations
mcp__memory__add_observations
mcp__memory__delete_entities
mcp__memory__delete_observations
mcp__memory__delete_relations
mcp__memory__read_graph
mcp__memory__search_nodes
mcp__memory__open_nodes
mcp__fetch__fetch
```

