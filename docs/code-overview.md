# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run`, `trace` и `skills`. |
| `madharness_mini/config.py` | Загружает настройки из defaults, локального конфига, `.env` и окружения. |
| `madharness_mini/context/` | Собирает prompt fragments, задачу пользователя, историю и бюджет контекста. |
| `madharness_mini/skills/` | Ищет skills, строит catalog, активирует `SKILL.md` и описывает resources. |
| `madharness_mini/loop.py` | Собирает запуск: trace, model client, skills discovery, context и registry. |
| `madharness_mini/model_loop.py` | Выполняет общий цикл model/tool calls. |
| `madharness_mini/model.py` | Вызывает OpenAI-совместимый `/chat/completions`. |
| `madharness_mini/tools/` | Описывает встроенные инструменты и provider `activate_skill`. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, защищённые пути и shell-команды. |
| `madharness_mini/instructions.py` | Загружает встроенный prompt и проектные инструкции `AGENTS.md`. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и краткую сводку. |

## Поток `madharness-mini run`

1. CLI создаёт `Config`.
2. `run_agent()` создаёт `Trace`, `ModelClient` и запускает discovery skills.
3. Если пользователь явно указал skill, он активируется до первого model call.
4. Если явного выбора нет, в контекст добавляется compact catalog, а в registry
   появляется tool `activate_skill`.
5. `base_context()` добавляет встроенный prompt и `AGENTS.md`.
6. `ToolRegistry` регистрирует встроенные инструменты и skill provider.
7. `ContextManager.messages()` применяет бюджет и отдаёт messages.
8. `model_loop.py` отправляет messages и tool schemas в модель.
9. Tool observation записывается в историю; служебный эффект активации skill
   добавляет durable fragment отдельно от observation.
10. Цикл продолжается до финального ответа или лимита `max_turns`.

## Границы skills

Skill не является плагином кода и не запускается сам. Он добавляет инструкции и
указывает на ресурсы внутри workspace. Если skill просит прочитать reference или
запустить script, модель всё равно должна использовать обычные инструменты:
`read_file` или `run_shell`.

Это делает skills хорошей учебной ступенью: студент видит, как расширяется
контекст, не смешивая это с внешними процессами или новыми протоколами.

## Тестовое покрытие

Основные сценарии лежат в `tests/test_skills.py` и соседних файлах: discovery,
frontmatter, explicit selection, compact catalog, activation, resources,
контекст и trace.
