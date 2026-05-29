# madharness-mini

> Учебная ветка: `06-subagents`
>
> Тема главы: markdown-субагенты, роли и оркестрация внутри одного harness.
>
> В этой точке parent agent может получить инструмент `delegate_task`, запускать
> встроенных или project-local субагентов, передавать им ограниченный набор
> tools и получать отдельные дочерние trace-файлы.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Предыдущая ветка: `05-mcp`
> Следующая ветка: `07-hooks`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Эта ветка показывает, как один
agent loop можно превратить в маленькую оркестрацию ролей, не добавляя внешних
runtime-зависимостей.

Проект написан для Python 3.13+ и использует OpenAI-совместимый API
`/chat/completions`.

## Что есть в этой ветке

- команды `init`, `ask`, `run`, `trace`, `skills` и `subagents`;
- проектные инструкции `AGENTS.md`;
- слой контекста с бюджетом и `context_report`;
- project-local Agent Skills и `activate_skill`;
- stdio MCP tools через `.madharness-mini/mcp.json`;
- встроенные роли `researcher`, `planner`, `implementer`, `reviewer`;
- project-local субагенты в `.madharness-mini/subagents/<name>.md`;
- режимы оркестрации `off`, `requested`, `auto`, `required`;
- инструменты `delegate_task` и, для разрешённых ролей, `ask_user`;
- отдельные дочерние traces для запусков субагентов.

В этой ветке ещё нет hooks. Здесь важны роли, profiles, allow-list tools и
передача результата обратно parent agent.

## Быстрый запуск

Установите `madharness-mini` из GitHub:

```bash
uv tool install madharness-mini --from git+https://github.com/MADTeacher/madharness-mini.git
```

Перейдите в корень продукта и создайте локальную настройку:

```bash
madharness-mini init \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --api-key "ключ-доступа-openrouter"
```

## Оркестрация

По умолчанию используется режим `auto`: parent agent видит `delegate_task`, но
может решить небольшую задачу сам.

Для одного запуска режим можно выбрать флагом:

```bash
madharness-mini run --no-orchestrate "Обнови короткий текст"
madharness-mini run --orchestration requested "Используй субагентов для ревью"
madharness-mini run --orchestrate-required "Разбей задачу между ролями"
```

В `.env` тот же выбор задаётся так:

```text
MADHARNESS_MINI_ORCHESTRATION_MODE=requested
```

Проверить встроенные и project-local роли:

```bash
madharness-mini subagents list
madharness-mini subagents validate
```

## Project-local субагент

Создайте `.madharness-mini/subagents/test-writer.md`:

```md
---
name: test-writer
description: Пишет минимальные unittest для изменённого кода.
profile: writable
tools: ["list_files", "read_file", "search_code", "apply_patch", "write_file", "run_shell", "ask_user"]
max_turns: 10
---

Ты субагент test-writer.
Пиши маленькие тесты рядом с существующими tests.
Если не хватает требований, задай один короткий вопрос через ask_user.
```

`profile` не выдаёт полномочия сам по себе. Фактический набор доступных tools
задаётся списком `tools`.

## Документация ветки

- [Возможности ветки](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Слой контекста](docs/context-layer.md)
- [Agent Skills](docs/agent-skills.md)
- [MCP](docs/mcp.md)
- [Субагенты](docs/subagents.md)
- [Инструмент apply_patch](docs/apply-patch.md)

## Разработка самого проекта

Если вы меняете код `madharness-mini`, запускайте проверки из корня этого
репозитория:

```bash
uv run -m unittest discover -s tests
```

Быстрая ручная проверка CLI:

```bash
uv run madharness-mini subagents list
uv run madharness-mini subagents validate
```

## Что дальше

Следующая ветка `07-hooks` добавляет lifecycle hooks: локальные команды проекта
смогут наблюдать события harness и блокировать tool call до выполнения.
