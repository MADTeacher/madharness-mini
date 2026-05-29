# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run`, `trace`, `skills` и `subagents`. |
| `madharness_mini/config.py` | Загружает настройки из defaults, локального конфига, `.env` и окружения. |
| `madharness_mini/context/` | Собирает prompt fragments, задачу пользователя, историю и бюджет контекста. |
| `madharness_mini/skills/` | Ищет skills, строит catalog, активирует `SKILL.md` и описывает resources. |
| `madharness_mini/mcp/` | Подключает stdio MCP tools через JSON-RPC. |
| `madharness_mini/subagents/` | Загружает роли, вычисляет режим оркестрации, запускает дочерний loop и tools субагентов. |
| `madharness_mini/hooks/` | Читает hooks config, запускает command handlers, делает redaction и пишет hook events в trace. |
| `madharness_mini/prompts/subagents/` | Хранит встроенные роли `researcher`, `planner`, `implementer`, `reviewer`. |
| `madharness_mini/loop.py` | Собирает запуск: trace, hooks, model client, skills, subagents, MCP, context и registry. |
| `madharness_mini/model_loop.py` | Выполняет общий цикл model/tool calls и вызывает hooks вокруг model/tool событий. |
| `madharness_mini/model.py` | Делает HTTP-запрос в OpenAI-совместимый `/chat/completions`. |
| `madharness_mini/tools/` | Описывает builtin tools и общий `ToolRegistry`. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, protected paths и shell-команды. |
| `madharness_mini/instructions.py` | Загружает встроенный prompt и проектные инструкции `AGENTS.md`. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и краткую сводку. |
| `madharness_mini/utils.py` | Хранит общие лимиты, helpers observation и JSON Schema. |

## Поток `madharness-mini run`

1. CLI создаёт `Config`.
2. `run_agent()` создаёт `Trace` и `HookManager`.
3. Hooks получают `session_start`.
4. Загружаются skills, subagents и MCP providers.
5. `ContextManager` собирает system fragments, `AGENTS.md`, catalog skills и историю.
6. Перед model call вызывается `before_model_call`, после ответа — `after_model_call`.
7. Если модель вызывает tool, `model_loop.py` вызывает `before_tool_call`.
8. Если hook блокирует действие, handler не запускается, а модель получает
   fail-observation.
9. Если блокировки нет, `ToolRegistry.call()` выполняет handler.
10. Tool observation записывается в trace и историю контекста.
11. `delegate_task` запускает дочерний loop субагента с отдельным trace.
12. MCP tools проксируются в stdio server через `tools/call`.
13. При нормальном завершении hooks получают `session_end`, при ошибке —
   `session_error`.

## Главные границы

Context layer не выполняет tools, а только готовит messages. Tool handlers не
должны сами решать, что модель увидит дальше, кроме обычного observation и
служебных эффектов вроде activation skill. MCP и hooks — внешние процессы, но
они запускаются явно, без shell-строки и без автоматической передачи секретов.

Субагент не наследует полный parent context и не получает `delegate_task`.
Hooks не заменяют `Policy`, а добавляют проектную проверку поверх неё.

## Тестовое покрытие

Основные сценарии покрыты в тематических файлах `tests/test_*.py`: config,
instructions, context, tools, model loop, skills, MCP, subagents, hooks и
policy.
