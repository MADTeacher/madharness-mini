# madharness-mini

> Учебная ветка: `01-minimalistic-harness`
>
> Тема главы: минимальный агентский harness для локального проекта.
> В этой точке уже есть CLI, OpenAI-совместимый model call, базовые инструменты
> работы с workspace, политика безопасности и JSONL-трассы.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Следующая ветка: `02-AGENTS-md`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Он даёт модели простой цикл:
получить задачу, вызвать инструмент, увидеть observation, продолжить работу и
записать ход выполнения в trace.

Проект написан для Python 3.13+ и не имеет runtime-зависимостей. Внутри
используется OpenAI-совместимый API `/chat/completions`, поэтому можно
подключить OpenRouter, KodikRouter, локальный совместимый сервер или другой
сервис с тем же форматом API.

## Что есть в этой ветке

- команда `init` для локального `.madharness-mini/config.json`;
- команда `ask` для одного запроса к модели без инструментов;
- команда `run` для агентского цикла с tool calls;
- команда `trace` для краткой сводки JSONL-трассы;
- инструменты `list_files`, `read_file`, `write_file`, `search_code`,
  `apply_patch` и `run_shell`;
- проверка workspace-границ, защищённых путей и явно опасных shell-команд;
- единый формат observation через `ok=false/true`;
- тесты для базового поведения harness.

В этой ветке ещё нет `AGENTS.md`, чтения изображений, отдельного слоя контекста,
Agent Skills, MCP, субагентов и hooks. Эти темы появляются в следующих ветках.

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

Перейдите в корень продукта, с которым должен работать агент:

```bash
cd /path/to/your/product
```

Создайте локальную настройку:

```bash
madharness-mini init \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --api-key "ключ-доступа-openrouter"
```

Команда создаёт `.madharness-mini/config.json`. В этом файле можно настроить
`model`, `base_url`, `api_key`, `temperature`, `max_turns`, `workspace_root`,
`protected_paths` и `allow_shell`.

## Первые команды

Задать вопрос без доступа к инструментам:

```bash
madharness-mini ask "Объясни, что делает этот проект"
```

Запустить агентский режим с чтением файлов, поиском, правками и проверками:

```bash
madharness-mini run "Найди команду для запуска тестов и объясни, что она проверяет"
```

Посмотреть краткую сводку трассы после запуска:

```bash
madharness-mini trace 20260529-171000
```

Идентификатор трассы берётся из строки `Trace: ...`, которую `ask` и `run`
печатают после ответа.

## Документация ветки

- [Возможности минимального harness](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
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

Локально переустановить команду можно так:

```bash
uv tool install --python 3.13 --force .
```

## Что дальше

Следующая ветка `02-AGENTS-md` добавляет проектные инструкции `AGENTS.md` и
первый мультимодальный инструмент `read_image`.

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
