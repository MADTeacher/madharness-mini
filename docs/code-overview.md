# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run` и `trace`. |
| `madharness_mini/config.py` | Загружает настройки из значений по умолчанию, локального конфига, `.env` и переменных окружения. |
| `madharness_mini/loop.py` | Управляет режимами `ask` и `run`, хранит историю сообщений и вызывает инструменты. |
| `madharness_mini/model.py` | Делает HTTP-запрос в OpenAI-совместимый `/chat/completions`. |
| `madharness_mini/tools/` | Описывает tool schemas и handlers для файлов, поиска, shell и patch. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, защищённые пути и shell-команды. |
| `madharness_mini/instructions.py` | Загружает встроенный системный prompt из `prompts/system.md`. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и строит краткую сводку. |
| `madharness_mini/utils.py` | Хранит общие лимиты, helpers для observation и JSON Schema. |

## Поток `madharness-mini run`

1. `cli.py` читает аргументы и создаёт `Config`.
2. `Config` применяет локальный конфиг, `.env` и переменные окружения.
3. `run_agent()` создаёт `Trace`, `ModelClient`, `ToolRegistry` и стартовые сообщения.
4. Модель получает `messages` и schemas доступных инструментов.
5. Если модель отвечает текстом, запуск завершается.
6. Если модель вызывает tool, `ToolRegistry` находит handler и выполняет его.
7. Handler проверяет путь или команду через `Policy`.
8. Observation записывается в trace и возвращается модели как `role=tool`.
9. Цикл продолжается до финального ответа или лимита `max_turns`.

## Где смотреть при лабораторных

- CLI и пользовательский вывод: `madharness_mini/cli.py`.
- Настройки и значения по умолчанию: `madharness_mini/config.py` и `utils.py`.
- Агентский цикл: `madharness_mini/loop.py`.
- Новые инструменты: `madharness_mini/tools/`.
- Безопасность путей и shell: `madharness_mini/policy.py`.
- Проверки trace: `madharness_mini/trace.py`.

Эта ветка намеренно держит всё близко к основному циклу. Следующие ветки будут
выносить проектные инструкции, контекст и расширения в отдельные слои.
