# Возможности ветки

`07-hooks` — финальная учебная точка маршрута. Здесь собраны все механизмы
предыдущих веток и добавлен проектный слой lifecycle hooks.

## Команды CLI

| Команда | Что делает |
| --- | --- |
| `madharness-mini init` | Создаёт или обновляет `.madharness-mini/config.json`. |
| `madharness-mini ask "..."` | Отправляет один запрос модели без tools. |
| `madharness-mini run "..."` | Запускает agent loop с tools, context, skills, MCP, subagents и hooks. |
| `madharness-mini trace <id>` | Показывает краткую сводку JSONL-трассы. |
| `madharness-mini skills list/show/validate` | Диагностирует project-local Agent Skills. |
| `madharness-mini subagents list/show/validate` | Диагностирует встроенных и project-local субагентов. |

## Hooks

Hooks настраиваются в:

```text
.madharness-mini/hooks.json
```

Hook — это локальная command handler, которая получает JSON-событие в stdin.
События: `session_start`, `before_model_call`, `after_model_call`,
`before_tool_call`, `after_tool_call`, `session_end`, `session_error`.

Только `before_tool_call` может остановить действие. Для блокировки hook
возвращает JSON:

```json
{ "ok": false, "block": "причина блокировки" }
```

Harness не запускает tool handler и возвращает модели обычное observation
`ok=false`.

## Остальные механизмы

Ветка сохраняет:

- `AGENTS.md` как проектные инструкции;
- context budget и `context_report`;
- `read_image` для vision input;
- Agent Skills и `activate_skill`;
- stdio MCP tools;
- markdown-субагентов, `delegate_task`, `ask_user` и дочерние traces;
- базовые workspace tools: `list_files`, `read_file`, `search_code`,
  `write_file`, `apply_patch`, `run_shell`.

## Безопасность hooks

- command и args запускаются без shell;
- `cwd` должен быть внутри workspace;
- payload обрезается и проходит redaction очевидных секретов;
- `MADHARNESS_MINI_*` не наследуются hook-командой автоматически;
- ошибки audit hooks пишутся в trace и обычно не ломают запуск;
- блокировка имеет смысл только для `before_tool_call`.

Hooks не заменяют `Policy`. Они добавляют проектные правила поверх общей защиты
workspace, shell-команд и protected paths.
