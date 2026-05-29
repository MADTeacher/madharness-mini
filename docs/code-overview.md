# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run` и `trace`. |
| `madharness_mini/config.py` | Собирает настройки из defaults, `.madharness-mini/config.json`, `.env` и окружения. |
| `madharness_mini/instructions.py` | Загружает встроенный prompt и проектные инструкции `AGENTS.md`. |
| `madharness_mini/loop.py` | Склеивает system prompt, задачу пользователя и историю tool calls. |
| `madharness_mini/model.py` | Вызывает OpenAI-совместимый `/chat/completions`. |
| `madharness_mini/tools/` | Хранит tool schemas и handlers, включая `read_image`. |
| `madharness_mini/policy.py` | Проверяет workspace-границы, защищённые пути и shell-команды. |
| `madharness_mini/trace.py` | Пишет JSONL-трассу и строит краткую сводку. |

## Поток `ask`

1. CLI создаёт `Config`.
2. `base_messages()` берёт встроенный system prompt.
3. `load_project_instructions()` добавляет найденные `AGENTS.md`.
4. `ModelClient` отправляет один запрос без tools.
5. Ответ и trace возвращаются пользователю.

## Поток `run`

1. `run_agent()` создаёт `Trace`, `ModelClient`, `ToolRegistry` и стартовые сообщения.
2. Модель получает сообщения и schemas инструментов.
3. Если модель вызывает tool, registry выполняет handler.
4. Observation пишется в trace и возвращается модели как `role=tool`.
5. Для `read_image` handler может добавить follow-up message с image payload.
6. Цикл завершается финальным текстом или лимитом `max_turns`.

## На что обратить внимание

`AGENTS.md` сейчас добавляется в system prompt простой строковой склейкой. Это
хорошо для учебной ветки, но быстро становится тесно: prompt, история,
изображения и дополнительные источники контекста начинают конкурировать за место. Именно эту
проблему решает ветка `03-Context-Layer`.
