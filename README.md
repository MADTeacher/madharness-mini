# madharness-mini

> Учебная ветка: `03-Context-Layer`
>
> Тема главы: слой контекста между задачей пользователя, проектными
> инструкциями, историей tool calls и запросом к модели.
>
> В этой точке harness уже отделяет сборку `messages` от основного model/tool
> loop, считает примерный бюджет контекста и пишет `context_report` в trace.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Предыдущая ветка: `02-AGENTS-md`
> Следующая ветка: `04-Agents-Skills`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Он даёт модели понятный цикл:
получить задачу, увидеть системные и проектные инструкции, вызвать инструмент,
получить observation и продолжить работу в рамках управляемого контекста.

Проект написан для Python 3.13+ и не имеет runtime-зависимостей. Внутри
используется OpenAI-совместимый API `/chat/completions`.

## Что есть в этой ветке

- команды `init`, `ask`, `run` и `trace`;
- проектные инструкции `AGENTS.md`;
- инструмент `read_image` для моделей с vision input;
- базовые инструменты workspace: `list_files`, `read_file`, `write_file`,
  `search_code`, `apply_patch`, `run_shell`;
- пакет `madharness_mini.context` с `ContextManager`,
  `ContextFragment` и `ContextProvider`;
- бюджет контекста через `context_max_tokens` и `context_keep_recent_turns`;
- `context_report` в trace перед каждым model call;
- общий `model_loop.py`, который отделяет цикл вызовов модели от публичных
  режимов `ask` и `run`.

В этой ветке ещё нет Agent Skills, MCP, субагентов и hooks. Здесь важно понять
саму механику контекста, потому что следующие ветки будут добавлять новые
источники инструкций и инструментов.

## Быстрый запуск

Установите `madharness-mini` из GitHub:

```bash
uv tool install madharness-mini --from git+https://github.com/MADTeacher/madharness-mini.git
```

Если терминал после установки не видит команду, обновите shell path и откройте
терминал заново:

```bash
uv tool update-shell
```

Перейдите в корень продукта, с которым должен работать агент, и создайте
локальную настройку:

```bash
madharness-mini init \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --api-key "ключ-доступа-openrouter"
```

Команда создаёт `.madharness-mini/config.json`. В этой ветке особенно важны
поля `context_max_tokens` и `context_keep_recent_turns`: они управляют
приблизительным бюджетом запроса к модели.

## Первые команды

Задать вопрос без доступа к инструментам:

```bash
madharness-mini ask "Объясни, что делает этот проект"
```

Запустить агентский режим:

```bash
madharness-mini run "Найди команду для запуска тестов и объясни, что она проверяет"
```

Посмотреть краткую сводку trace:

```bash
madharness-mini trace 20260529-171000
```

В trace этой ветки у событий `model_call_started` есть `context_report`. Он
показывает примерный размер messages, tools, список фрагментов и факт обрезки
истории или tool outputs.

## Документация ветки

- [Возможности ветки](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Слой контекста](docs/context-layer.md)
- [Инструмент apply_patch](docs/apply-patch.md)

## Разработка самого проекта

Если вы меняете код `madharness-mini`, запускайте проверки из корня этого
репозитория:

```bash
uv run -m unittest discover -s tests
```

Быстрая ручная проверка CLI:

```bash
uv run madharness-mini ask "Объясни, что делает этот проект"
```

## Что дальше

Следующая ветка `04-Agents-Skills` использует слой контекста, чтобы подключать
project-local `SKILL.md` как отдельные инструкции, а не как часть монолитного
system prompt.

## Лицензирование

Проект использует раздельную лицензионную модель:

- код распространяется по PolyForm Noncommercial License 1.0.0;
- учебные материалы распространяются по Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International.

Некоммерческое самообучение, академическое преподавание и исследовательское
использование разрешены на условиях соответствующих лицензий. Коммерческое
использование кода или материалов требует предварительного письменного
разрешения правообладателя.

Полные тексты и русские версии: [LICENSE.md](LICENSE.md).
