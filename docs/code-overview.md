# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run`, `trace` и `skills`. |
| `madharness_mini/config.py` | Загружает настройки из defaults, локального конфига, `.env` и окружения. |
| `madharness_mini/context/` | Собирает prompt fragments, задачу пользователя, историю и бюджет контекста. |
| `madharness_mini/skills/` | Ищет skills, строит catalog, активирует `SKILL.md` и описывает resources. |
| `madharness_mini/mcp/` | Реализует stdio MCP client, JSON-RPC, config loader, provider и result adapter. |
| `madharness_mini/loop.py` | Собирает запуск: trace, model client, skills discovery, MCP providers, context и registry. |
| `madharness_mini/model_loop.py` | Выполняет общий цикл model/tool calls. |
| `madharness_mini/tools/` | Описывает встроенные tools и общий `ToolRegistry`. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, защищённые пути и shell-команды. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и краткую сводку. |

## Поток `madharness-mini run`

1. CLI создаёт `Config`.
2. `run_agent()` создаёт `Trace`, `ModelClient` и запускает discovery skills.
3. Skill catalog или явно выбранный skill добавляется в context.
4. `base_context()` добавляет встроенный prompt и `AGENTS.md`.
5. `ToolRegistry` получает встроенные tools, `activate_skill` и MCP tools из
   `.madharness-mini/mcp.json`, если файл есть.
6. MCP provider запускает включённые stdio servers, вызывает `initialize` и
   `tools/list`, затем создаёт `ToolSpec` для каждого внешнего tool.
7. `ContextManager.messages()` применяет бюджет.
8. `model_loop.py` отправляет messages и tool schemas модели.
9. При MCP tool call provider отправляет `tools/call` внешнему серверу и
   превращает ответ в обычное observation.
10. `ToolRegistry.close()` закрывает MCP-процессы в `finally`.

## Границы MCP

MCP-сервер — доверенный локальный процесс, но harness всё равно задаёт явные
границы: запуск без shell, workspace-relative `cwd`, безопасное окружение и
обычный формат observation. Это учебно важно: внешний tool не должен ломать
модельный протокол и не должен незаметно получать секреты API.

## Тестовое покрытие

Основные сценарии MCP лежат в `tests/test_mcp.py`: config, stdio protocol,
provider, result conversion и закрытие процессов. Остальные тесты продолжают
проверять config, context, skills, policy, tools и model loop.
