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
| `madharness_mini/prompts/subagents/` | Хранит встроенные роли `researcher`, `planner`, `implementer`, `reviewer`. |
| `madharness_mini/loop.py` | Собирает запуск: trace, model client, skills, subagents, MCP, context и registry. |
| `madharness_mini/model_loop.py` | Выполняет общий цикл model/tool calls. |
| `madharness_mini/tools/` | Описывает встроенные tools и общий `ToolRegistry`. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, защищённые пути и shell-команды. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и краткую сводку. |

## Поток `madharness-mini run`

1. CLI создаёт `Config`.
2. `run_agent()` создаёт `Trace`, `ModelClient`, discovery skills и loader субагентов.
3. `subagents/orchestration.py` вычисляет режим: `off`, `requested`, `auto` или `required`.
4. Skill catalog или явно выбранный skill добавляется в context.
5. `base_context()` добавляет встроенный prompt и `AGENTS.md`.
6. `ToolRegistry` получает встроенные tools, `activate_skill`, `delegate_task`
   при разрешённой оркестрации и MCP tools из `mcp.json`.
7. Parent model loop работает как обычно.
8. При `delegate_task` запускается дочерний loop с отдельным context, role
   prompt, allow-list tools и локальным trace.
9. Результат субагента возвращается parent как observation.
10. Если субагент вызвал `ask_user`, основной `run` печатает вопрос и завершается.

## Границы субагентов

Субагент не наследует полную историю parent. Он получает задачу делегации,
краткий parent context, свой markdown prompt и только разрешённые tools.

Встроенный `planner` дополнительно ограничен `.md` файлами, чтобы планирование
не превращалось в реализацию. Субагент не получает `delegate_task`, иначе
оркестрация могла бы стать рекурсивной и плохо наблюдаемой.

## Тестовое покрытие

Основные сценарии лежат в `tests/test_subagents.py`: loader, validation,
orchestration modes, delegation, role tools, `ask_user` и дочерние traces.
