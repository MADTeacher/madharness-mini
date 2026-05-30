# Возможности полной версии

`madharness-mini` помогает модели работать с локальным репозиторием через
управляемый харнесс: контекст, инструменты, безопасность, трассы и расширения
остаются явно описанными в коде.

## Команды CLI

| Команда | Что делает |
| --- | --- |
| `madharness-mini init` | Создаёт или обновляет `.madharness-mini/config.json`. |
| `madharness-mini ask "..."` | Отправляет один запрос модели без инструментов. |
| `madharness-mini run "..."` | Запускает агентский цикл с инструментами, контекстом, навыками, MCP, субагентами и hooks. |
| `madharness-mini trace <id>` | Показывает краткую сводку JSONL-трассы. |
| `madharness-mini skills list/show/validate` | Диагностирует проектные Agent Skills. |
| `madharness-mini subagents list/show/validate` | Диагностирует встроенных и проектных субагентов. |

## Основные механизмы

| Механизм | Где описан |
| --- | --- |
| Проектные инструкции `AGENTS.md` | `madharness_mini/instructions.py` |
| Слой контекста и бюджет | [context-layer.md](context-layer.md) |
| Встроенные workspace-инструменты | `madharness_mini/tools/` |
| Agent Skills | [agent-skills.md](agent-skills.md) |
| MCP-инструменты | [mcp.md](mcp.md) |
| Субагенты | [subagents.md](subagents.md) |
| Hooks | [hooks.md](hooks.md) |
| Трасса | `madharness_mini/trace.py` |

## Инструменты `run`

| Инструмент | Что делает |
| --- | --- |
| `list_files` | Показывает файлы внутри workspace с лимитом результатов. |
| `read_file` | Читает UTF-8 фрагмент файла по строкам. |
| `read_image` | При включённом vision input прикрепляет локальное изображение к следующему обращению к модели. |
| `search_code` | Ищет буквальную подстроку в файлах проекта. |
| `apply_patch` | Применяет точечные файловые правки. |
| `write_file` | Полностью перезаписывает UTF-8 файл. |
| `run_shell` | Запускает одну разрешённую команду в workspace или безопасном `cwd`. |
| `activate_skill` | Подключает проектный навык в закреплённый контекст. |
| `delegate_task` | Делегирует задачу Markdown-субагенту. |
| `ask_user` | Доступен только разрешённым субагентам; завершает запуск вопросом пользователю. |
| `mcp__...` | Инструменты, полученные от явно включённых stdio MCP-серверов. |

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
- payload для hooks обрезается и проходит redaction;
- субагенты получают только разрешённые инструменты.

## Трасса

Каждый `ask` и `run` пишет JSONL-трассу в `.madharness-mini/traces`. В трассе
видны обращения к модели, ответы инструментов, отчёты о контексте, активации
навыков, жизненный цикл MCP, события субагентов и hooks. Это главный учебный
источник для разбора того, что агент реально сделал.
