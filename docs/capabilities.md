# Возможности ветки

`05-mcp` добавляет внешние tools через минимальный stdio MCP client. Для модели
MCP-инструмент выглядит как обычный tool, а для harness это отдельный provider,
который управляет процессом и JSON-RPC.

## Команды CLI

| Команда | Что делает |
| --- | --- |
| `madharness-mini init` | Создаёт или обновляет `.madharness-mini/config.json`. |
| `madharness-mini ask "..."` | Отправляет один запрос модели без tools. |
| `madharness-mini run "..."` | Запускает agent loop с базовыми tools, skills и MCP. |
| `madharness-mini trace <id>` | Показывает краткую сводку JSONL-трассы. |
| `madharness-mini skills list/show/validate` | Диагностирует project-local Agent Skills. |

## MCP tools

Настройки MCP лежат отдельно:

```text
.madharness-mini/mcp.json
```

Если файла нет, MCP выключен. Harness запускает только серверы с
`enabled=true`. Поддерживается stdio transport и tools API: `initialize`,
`tools/list`, `tools/call`.

MCP tool получает имя:

```text
mcp__<server>__<tool>
```

Так модель видит источник инструмента, а registry избегает конфликтов имён.

## Безопасность MCP

- `command` и `args` запускаются списком, без shell;
- `cwd` должен быть внутри workspace;
- `MADHARNESS_MINI_*` и ключ модели не наследуются автоматически;
- секреты передаются MCP-серверу только через явный `env`;
- результат MCP приводится к обычному observation и обрезается по общим лимитам;
- provider закрывается через `ToolRegistry.close()`.

## Остальные возможности

Ветка сохраняет всё из предыдущих глав: `AGENTS.md`, context budget, `read_image`,
Agent Skills, `activate_skill`, базовые файловые tools, `apply_patch` и
`run_shell`.

## Что не входит в эту ветку

Здесь ещё нет субагентов и hooks. MCP расширяет набор tools, но не добавляет
отдельные роли, делегацию или lifecycle-политику вокруг каждого tool call.
