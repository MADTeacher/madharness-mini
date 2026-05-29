# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run` и `trace`. |
| `madharness_mini/config.py` | Собирает настройки из defaults, локального конфига, `.env` и окружения. |
| `madharness_mini/context/` | Собирает prompt fragments, задачу пользователя, историю и бюджет контекста. |
| `madharness_mini/loop.py` | Остаётся публичным фасадом режимов `ask` и `run`. |
| `madharness_mini/model_loop.py` | Выполняет общий цикл model/tool calls и работу с observation. |
| `madharness_mini/model.py` | Отправляет HTTP-запрос в OpenAI-совместимый `/chat/completions`. |
| `madharness_mini/tools/` | Описывает встроенные инструменты и их handlers. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, защищённые пути и shell-команды. |
| `madharness_mini/instructions.py` | Загружает встроенный prompt и проектные инструкции `AGENTS.md`. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и строит краткую сводку. |
| `madharness_mini/utils.py` | Хранит общие лимиты, helpers observation и JSON Schema. |

## Поток `madharness-mini run`

1. `cli.py` читает аргументы и создаёт `Config`.
2. `run_agent()` создаёт `Trace`, `ModelClient`, `ToolRegistry` и `ContextManager`.
3. `base_context()` добавляет встроенный prompt и найденные `AGENTS.md` как
   закреплённые фрагменты.
4. `ContextManager.messages()` собирает system/user/history и применяет бюджет.
5. `model_loop.py` отправляет messages и tool schemas в модель.
6. Если модель отвечает текстом, запуск завершается.
7. Если модель вызывает tool, `ToolRegistry` выполняет handler.
8. Observation записывается в trace и в историю контекста.
9. Цикл продолжается до финального ответа или лимита `max_turns`.

## Данные и границы

`Config.root` задаёт workspace. Все файловые инструменты проходят через
`Policy.safe_path()`, поэтому относительные пути не выходят за рабочую папку, а
защищённые пути из `protected_paths` блокируются.

Ответы инструментов имеют единый вид: успешные создаются через `ok()`,
ошибочные через `fail()`. Модель получает observation с полями `ok`, `tool` и
`summary`, а дополнительные данные зависят от конкретного инструмента.

Слой контекста не вызывает handlers и не проверяет безопасность путей. Его роль
уже: хранить то, что будет отправлено модели, и объяснять через `context_report`,
почему именно этот набор messages поместился в запрос.

## Тестовое покрытие

Основные сценарии лежат в `tests/test_*.py`: конфигурация, проектные
инструкции, policy, инструменты, model loop и слой контекста.
