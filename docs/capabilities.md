# Возможности полной версии

`madharness-mini` помогает модели работать с локальным репозиторием через
управляемый harness: контекст, инструменты, безопасность, трассы и расширения
остаются явно описанными в коде.

## Команды CLI

| Команда | Что делает |
| --- | --- |
| `madharness-mini init` | Создаёт или обновляет `.madharness-mini/config.json`. |
| `madharness-mini ask "..."` | Отправляет один запрос модели без tools. |
| `madharness-mini run "..."` | Запускает agent loop с tools, context, skills, MCP, subagents и hooks. |
| `madharness-mini trace <id>` | Показывает краткую сводку JSONL-трассы. |
| `madharness-mini skills list/show/validate` | Диагностирует project-local Agent Skills. |
| `madharness-mini subagents list/show/validate` | Диагностирует встроенных и project-local субагентов. |

## Основные механизмы

| Механизм | Где описан |
| --- | --- |
| Проектные инструкции `AGENTS.md` | `madharness_mini/instructions.py` |
| Слой контекста и бюджет | [context-layer.md](context-layer.md) |
| Builtin tools workspace | `madharness_mini/tools/` |
| Agent Skills | [agent-skills.md](agent-skills.md) |
| MCP tools | [mcp.md](mcp.md) |
| Субагенты | [subagents.md](subagents.md) |
| Hooks | [hooks.md](hooks.md) |
| Trace | `madharness_mini/trace.py` |

## Инструменты `run`

| Инструмент | Что делает |
| --- | --- |
| `list_files` | Показывает файлы внутри workspace с лимитом результатов. |
| `read_file` | Читает UTF-8 фрагмент файла по строкам. |
| `read_image` | При включённом vision input прикрепляет локальное изображение к следующему model call. |
| `search_code` | Ищет буквальную подстроку в файлах проекта. |
| `apply_patch` | Применяет точечные файловые правки. |
| `write_file` | Полностью перезаписывает UTF-8 файл. |
| `run_shell` | Запускает одну разрешённую команду в workspace или безопасном `cwd`. |
| `activate_skill` | Подключает project-local skill в durable context. |
| `delegate_task` | Делегирует задачу markdown-субагенту. |
| `ask_user` | Доступен только разрешённым субагентам; завершает запуск вопросом пользователю. |
| `mcp__...` | Tools, полученные от явно включённых stdio MCP-серверов. |

## Настройки

Основной файл:

```text
.madharness-mini/config.json
```

Отдельные файлы расширений:

```text
.madharness-mini/mcp.json
.madharness-mini/hooks.json
.madharness-mini/subagents/<name>.md
.madharness_mini/skills/<name>/SKILL.md
.agents/skills/<name>/SKILL.md
```

Важные поля `config.json`: `model`, `base_url`, `api_key`, `temperature`,
`max_turns`, `context_max_tokens`, `context_keep_recent_turns`,
`orchestration_mode`, `subagent_max_turns`, `subagent_context_max_tokens`,
`workspace_root`, `protected_paths`, `allow_shell`, `supports_image_input`,
`max_image_bytes`, `image_detail`.

## Безопасность

- файловые инструменты проходят через `Policy.safe_path()`;
- protected paths блокируются даже внутри workspace;
- `run_shell` запрещает shell-цепочки, редиректы и явно рискованные команды;
- MCP и hooks запускаются без shell-строки;
- секреты модели не передаются MCP и hooks автоматически;
- hooks проходят redaction payload;
- subagents получают только разрешённые tools.

## Trace

Каждый `ask` и `run` пишет JSONL trace в `.madharness-mini/traces`. В trace
видны model calls, tool observations, context reports, activation skills,
MCP lifecycle, subagent events и hook events. Это главный учебный источник для
разбора того, что агент реально сделал.
