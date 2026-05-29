# Структура кода

## Основные модули

| Модуль | За что отвечает |
| --- | --- |
| `madharness_mini/cli.py` | Разбирает команды `init`, `ask`, `run`, `trace`, `skills` и `subagents`, печатает результат пользователю. |
| `madharness_mini/config.py` | Собирает настройки из значений по умолчанию, `config.json`, `.env` и переменных окружения. |
| `madharness_mini/context/` | Собирает сообщения для модели: системные фрагменты, задачу пользователя, историю assistant/tool, bootstrap стартового контекста и бюджет. |
| `madharness_mini/loop.py` | Остаётся фасадом `ask`/`run`: собирает запуск, discovery расширений и registry инструментов. |
| `madharness_mini/model_loop.py` | Выполняет общий цикл model/tool вызовов, retry при коротком rate limit и служебные эффекты observation. |
| `madharness_mini/model.py` | Отправляет HTTP-запрос в OpenAI-совместимый `/chat/completions`. |
| `madharness_mini/tools/` | Описывает инструменты агента, регистрирует их и выполняет операции с файлами, поиском, patch и shell-командами. |
| `madharness_mini/skills/` | Ищет project-local Agent Skills, строит каталог, активирует `SKILL.md` как контекстный фрагмент и отдаёт provider инструмента `activate_skill`. |
| `madharness_mini/subagents/` | Загружает markdown-субагентов, добавляет `delegate_task` и `ask_user`, вычисляет режим оркестрации, запускает дочерний loop и строит сводку локальных trace-файлов. |
| `madharness_mini/mcp/` | Минимальный stdio MCP-клиент: отдельный `mcp.json`, JSON-RPC, subprocess transport, adapter в `ToolSpec`. |
| `madharness_mini/policy.py` | Проверяет, что инструменты не выходят за workspace и не запускают явно рискованные команды. |
| `madharness_mini/instructions.py` | Загружает встроенный системный промпт и собирает проектные инструкции `AGENTS.md`. |
| `madharness_mini/prompts/subagents/` | Markdown-конфиги и системные prompt-файлы встроенных ролей `researcher`, `planner`, `implementer`, `reviewer`. |
| `madharness_mini/trace.py` | Записывает JSONL-журнал запуска и строит короткую сводку по результатам работы сессии. |
| `madharness_mini/utils.py` | Хранит общие константы, схемы параметров и формат ответов инструментов. |

## Поток выполнения `madharness-mini run`

1. `cli.main()` читает аргументы командной строки и создаёт `Config`.
2. `Config` загружает локальные настройки из `.madharness-mini/config.json`, затем применяет `.env` и переменные `MADHARNESS_MINI_*`.
3. `run_agent()` создаёт `Trace`, `ModelClient`, запускает discovery Agent Skills и загружает markdown-субагентов из `madharness_mini/prompts/subagents/*.md` и `.madharness-mini/subagents/*.md`.
4. `subagents/orchestration.py` вычисляет режим оркестрации (`off`, `requested`, `auto`, `required`) из конфига, `.env` и CLI-флагов; `run_agent()` пишет событие `orchestration_mode`.
5. Если пользователь явно указал skill, он активируется до первого model call; иначе в контекст добавляется compact catalog, а в registry — provider `activate_skill`.
6. `context/bootstrap.py` создаёт `ContextManager`, берёт системный промпт из `prompts/system.md` и добавляет найденные проектные инструкции `AGENTS.md` как закреплённый фрагмент.
7. `ToolRegistry` регистрирует встроенные инструменты, `activate_skill`, `delegate_task` при разрешённой оркестрации и MCP tools из `.madharness-mini/mcp.json`, если этот файл есть.
8. `ContextManager.messages()` собирает system/user/history сообщения и применяет символьный бюджет контекста.
9. `ModelClient.chat()` отправляет сообщения и схемы инструментов в LLM API.
10. Если модель вернула обычный текст, `model_loop.py` завершает запуск и пишет результат в трассу.
11. Если модель вернула `tool_calls`, `ToolRegistry.call()` находит `ToolSpec` по имени и вызывает handler инструмента.
12. Handler получает `ToolContext` с `Config`, `Policy`, трассой и runtime навыков, чтобы проверить путь, shell-команду, активацию skill или делегацию субагенту.
13. `ContextManager.record_assistant()` и `record_tool_result()` добавляют ответ модели и observation инструмента в историю. Служебные эффекты вроде нового context-фрагмента от `activate_skill` применяются отдельно и не попадают в observation.
14. Если модель вызывает `delegate_task`, `subagents/runner.py` запускает дочерний agent loop с собственным контекстом, allow-list tools из markdown-файла и локальным trace.
15. Цикл продолжается до финального ответа модели или до лимита `max_turns`.

## Данные и границы

`Config.root` задаёт рабочую папку агента. Все файловые инструменты сначала проходят через `Policy.safe_path()`, поэтому относительные пути превращаются в абсолютные только внутри workspace. Защищённые имена из `protected_paths` блокируются даже тогда, когда они находятся внутри workspace.

Ответы инструментов имеют единый вид: успешные создаются через `ok()`, ошибочные через `fail()`. Благодаря этому модель получает предсказуемое наблюдение с полями `ok`, `tool` и `summary`, а дополнительные данные зависят от конкретного инструмента.

Новые встроенные инструменты добавляются в пакет `madharness_mini/tools/`: handler и `ToolSpec` лежат рядом в смысловом модуле, а стандартный набор отдаёт `BuiltinToolProvider`. Agent Skills и MCP подключают дополнительные providers, поэтому внешний источник может добавить `ToolSpec` без правки агентского цикла.

Оркестрация подключается как обычный provider, когда текущий `orchestration_mode` разрешает `delegate_task`. В `auto` parent видит делегацию как ещё один tool, но может решить задачу сам. В `requested` tool появляется только при явном запросе на субагентов или оркестрацию. В `required` parent получает только read/search tools и `delegate_task`, поэтому выступает координатором, а правки и проверки поручает ролям. Субагент не наследует историю родителя: он получает задачу, краткий parent context, свой markdown prompt и tools из profile. Для встроенного `planner` registry добавляет write scope только на `.md` файлы, чтобы он вёл план, а не реализацию. `ask_user` доступен только тем субагентам, где он указан в `tools`; если он возвращает `needs_user_input`, основной `run` напрямую печатает вопрос пользователю и завершает запуск.

MCP включается отдельным файлом `.madharness-mini/mcp.json`. `madharness_mini.mcp.config` читает только явно включённые серверы, `stdio.py` запускает процесс без shell, `provider.py` префиксует имена tools как `mcp__server__tool`, а `results.py` приводит `tools/call` к обычному observation. `ToolRegistry.close()` вызывается в `run_agent()` через `finally`, чтобы stdio-процессы закрывались при успехе и ошибке.

Слой контекста отделён от инструментов. `madharness_mini.context` не вызывает handlers и не проверяет безопасность путей; он только хранит фрагменты и историю, которые будут отправлены модели. Каталог skills добавляется через `ContextProvider`, а активированный skill становится durable `ContextFragment`, защищённым от удаления старой истории.

Подробные документы по расширениям лежат отдельно:

- [Agent Skills в madharness-mini](agent-skills.md)
- [План поддержки Agent Skills](agent-skills-plan.html)
- [MCP в madharness-mini](mcp.md)
- [Субагенты](subagents.md)

Трассы лежат в `.madharness-mini/traces/*.jsonl`. Каждая строка - отдельное событие: старт сессии, вызов модели, ответ модели, наблюдение инструмента или итоговый результат. События `model_call_started` включают `context_report`, чтобы можно было увидеть токеновую оценку и состав контекста перед запросом. Команда `madharness-mini trace <id>` читает такой файл и показывает краткую сводку.

У субагента появляется свой дочерний trace-файл в том же каталоге. Родительская трасса пишет только `subagent_started`, `subagent_finished` или `subagent_failed` и ссылку на локальный trace, а полный ход дочернего агента остаётся в его JSONL.

## Тестовое покрытие

Основные сценарии покрыты в тематических файлах `tests/test_*.py`: загрузка конфигурации, переопределение через `.env`, политика путей и shell-команд, формат наблюдений инструментов и файловые правки.
