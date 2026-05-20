# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/__main__.py` | Точка входа `python -m madharness_mini`; делегирует в `cli.main()`. |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run` и `trace`, печатает результат пользователю. |
| `madharness_mini/config.py` | Собирает настройки из значений по умолчанию, `.madharness-mini/config.json`, `.env` и переменных `MADHARNESS_MINI_*`. |
| `madharness_mini/loop.py` | Управляет режимами `ask` и `run`: готовит сообщения, вызывает модель, обрабатывает tool calls и повторяет запрос при коротком HTTP 429. |
| `madharness_mini/model.py` | Отправляет HTTP-запрос в OpenAI-совместимый `/chat/completions`; выделяет `ModelRateLimitError` для повтора в `loop.py`. |
| `madharness_mini/tools/` | Пакет инструментов агента: схемы для API, диспетчер вызовов и handlers для файлов, поиска, patch и shell. |
| `madharness_mini/policy.py` | Проверяет, что инструменты не выходят за workspace и не запускают явно рискованные shell-команды. |
| `madharness_mini/instructions.py` | Загружает встроенный системный промпт из `prompts/` и собирает цепочку проектных инструкций `AGENTS.md`. |
| `madharness_mini/trace.py` | Записывает JSONL-журнал запуска и строит короткую сводку по результатам работы сессии. |
| `madharness_mini/utils.py` | Хранит общие константы, лимиты ответов инструментов, формат наблюдений (`ok`/`fail`) и заготовки JSON Schema. |

## Пакет `madharness_mini/tools/`

| Модуль | За что отвечает |
| --- | --- |
| `specs.py` | Dataclass `ToolSpec`: имя, описание, JSON Schema параметров и handler. |
| `context.py` | `ToolContext` с `Config` и `Policy` — общий контекст для всех handlers. |
| `registry.py` | `ToolRegistry`: регистрация встроенных инструментов, схемы для модели, безопасный вызов по имени. |
| `builtins.py` | Явный список `builtin_specs()` — единственное место подключения новых встроенных инструментов. |
| `files.py` | `list_files`, `read_file`, `write_file`, `replace_text`. |
| `search.py` | `search_code` — точный текстовый поиск по файлам workspace. |
| `patch.py` | `apply_patch` — Codex-style патчи (см. [apply-patch.md](apply-patch.md)). |
| `shell.py` | `run_shell` — одна простая команда в рабочей папке с проверкой политики. |

Автообнаружения и внешних плагинов нет: новый инструмент добавляют handler + `ToolSpec` в смысловой модуль и включают spec в `builtin_specs()`.

## Поток выполнения `madharness-mini ask`

1. `cli.main()` читает аргументы и создаёт `Config`.
2. `ask()` создаёт `Trace` с kind=`ask` и собирает `base_messages()`.
3. `base_messages()` берёт системный промпт из `prompts/system.md`, при необходимости дополняет цепочкой `AGENTS.md` и добавляет задачу пользователя.
4. `ModelClient.chat()` отправляет сообщения **без** схем инструментов.
5. При HTTP 429 с коротким `Retry-After` (до 60 с) `call_model_with_rate_limit_retry()` один раз ждёт и повторяет запрос; длинный лимит или повторная 429 — ошибка в CLI и в трассе.
6. Текст ответа записывается в трассу как `session_end`, путь к файлу трассы печатается в stderr.

## Поток выполнения `madharness-mini run`

1. `cli.main()` читает аргументы командной строки и создаёт `Config`.
2. `Config` загружает локальные настройки из `.madharness-mini/config.json`, затем применяет `.env` и переменные `MADHARNESS_MINI_*`; вычисляет абсолютный `root` из `workspace_root`.
3. `run_agent()` создаёт `Trace`, `ModelClient` и `ToolRegistry`.
4. `base_messages()` готовит стартовую историю так же, как в режиме `ask`.
5. На каждом ходе `ModelClient.chat()` отправляет сообщения и схемы инструментов; `parallel_tool_calls` принудительно выключен.
6. Если модель вернула обычный текст, `run_agent()` завершает запуск и пишет результат в трассу.
7. Если модель вернула `tool_calls`, для каждого вызова `parse_tool_args()` извлекает имя и аргументы, затем `ToolRegistry.call()` находит `ToolSpec` и вызывает handler.
8. Handler получает `ToolContext` с `Config` и `Policy`, чтобы проверить путь относительно проекта или shell-команду перед её запуском.
9. Результат инструмента сериализуется в JSON и добавляется в историю как сообщение роли `tool`.
10. Цикл продолжается до финального ответа модели, ошибки API или до лимита `max_turns`.

## Данные и границы

`Config.root` задаёт рабочую папку агента. Все файловые инструменты сначала проходят через `Policy.safe_path()`, поэтому относительные пути превращаются в абсолютные только внутри workspace. Защищённые имена из `protected_paths` блокируются даже тогда, когда они находятся внутри workspace.

`Policy.shell_allowed()` разрешает только одну простую команду без пайпов, редиректов и явно запрещённых фрагментов; `allow_shell=false` отключает `run_shell` целиком.

Ответы инструментов имеют единый вид: успешные создаются через `ok()`, ошибочные через `fail()`. Модель получает предсказуемое наблюдение с полями `ok`, `tool` и `summary`, а дополнительные данные зависят от конкретного инструмента. Слишком длинный вывод обрезается функцией `clipped()` до `MAX_OUTPUT` символов.

При листинге и поиске `ignored()` скрывает служебные каталоги (`.git`, `.madharness-mini`, `__pycache__`, `.venv`, `.uv-cache`).

Проектные инструкции читаются из `AGENTS.md` по цепочке папок от `workspace_root` до текущей `cwd`; общий объём ограничен 32 KiB. Поддерживается только точное имя `AGENTS.md`.

Трассы лежат в `.madharness-mini/traces/*.jsonl`. Каждая строка — отдельное событие: `session_start`, вызов модели, ответ модели, `tool_observation`, `model_rate_limit_retry` или `session_end`. Команда `madharness-mini trace <id>` читает файл по префиксу id и показывает краткую сводку.

## Тестовое покрытие

Основные сценарии покрыты в тематических файлах `tests/test_*.py`:

| Файл | Что проверяет |
| --- | --- |
| `test_config_cli.py` | Слияние defaults и `config.json`, игнор legacy-полей, переопределение через `.env`, команда `init`. |
| `test_instructions.py` | Загрузка системного промпта, цепочка `AGENTS.md`, лимит 32 KiB, игнор пустых и неподдерживаемых имён. |
| `test_model_loop.py` | Настройки `ModelClient`, разбор `Retry-After`, HTTP 429, повтор ask при коротком rate limit. |
| `test_policy_utils.py` | Границы workspace, `protected_paths`, политика shell, формат наблюдений, `parse_tool_args`. |
| `test_tools.py` | Файловые инструменты, `apply_patch` (создание, правка, удаление, move, ошибки контекста и политики). |

Общая база для временных workspace — `tests/helpers.py` (`HarnessTestCase`).

Запуск из корня репозитория:

```bash
uv run -m unittest discover -s tests
```
