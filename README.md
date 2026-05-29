# madharness-mini

`madharness-mini` — учебный минималистичный harness для курса, книги и
лабораторных работ по harness-инженерии кодирующих ИИ-агентов.

Репозиторий устроен как сквозной маршрут: каждая учебная ветка фиксирует
отдельную ступень развития harness, а внутри ветки лежат актуальные для этой
ступени `README.md`, `LABS.md` и `docs/`.

Финальное состояние на `main` совпадает по возможностям с полной версией
harness: project instructions, context layer, Agent Skills, MCP, субагенты и
hooks.

## Учебный маршрут

| Ветка | Тема | Главный вопрос |
| --- | --- | --- |
| `01-minimalistic-harness` | Минимальный harness | Как устроить базовый цикл model/tool/trace? |
| `02-AGENTS-md` | Проектные инструкции и изображения | Как добавить локальные правила проекта и vision input? |
| `03-Context-Layer` | Слой контекста | Что именно модель видит перед каждым вызовом? |
| `04-Agents-Skills` | Agent Skills | Как подключать workflow-инструкции без изменения ядра? |
| `05-mcp` | MCP tools | Как превратить внешний stdio MCP-сервер в обычные tools модели? |
| `06-subagents` | Субагенты | Как делегировать задачи ролям с отдельными tools и traces? |
| `07-hooks` | Hooks | Как добавить проектный аудит и блокировку действий? |

Подробная карта курса: [COURSE.md](COURSE.md).

В каждой учебной ветке:

- `README.md` объясняет, где вы находитесь и что умеет именно эта версия;
- `LABS.md` даёт задачи трёх уровней, без оценок времени;
- `docs/README.md` ведёт только к актуальным документам этой ветки.

## Быстрый старт финальной версии

Установите команду из GitHub:

```bash
uv tool install madharness-mini --from git+https://github.com/MADTeacher/madharness-mini.git
```

Если терминал после установки не видит `madharness-mini`, обновите shell path и
откройте терминал заново:

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

Задайте вопрос без инструментов:

```bash
madharness-mini ask "Объясни, что делает этот проект"
```

Запустите агентский режим:

```bash
madharness-mini run "Найди команду для запуска тестов и объясни, что она проверяет"
```

Посмотрите trace:

```bash
madharness-mini trace 20260529-171000
```

## Финальная документация

- [Возможности полной версии](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Слой контекста](docs/context-layer.md)
- [Agent Skills](docs/agent-skills.md)
- [MCP](docs/mcp.md)
- [Субагенты](docs/subagents.md)
- [Hooks](docs/hooks.md)
- [Инструмент apply_patch](docs/apply-patch.md)

## Разработка самого проекта

Проект написан для Python 3.13+ и не имеет runtime-зависимостей.

Проверка:

```bash
uv run -m unittest discover -s tests
```

Быстрая ручная проверка CLI:

```bash
uv run madharness-mini ask "Объясни, что делает этот проект"
```

Локальная переустановка команды:

```bash
uv tool install --python 3.13 --force .
```
