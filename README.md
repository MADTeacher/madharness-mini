# madharness-mini

> Учебная ветка: `04-Agents-Skills`
>
> Тема главы: project-local Agent Skills как управляемый способ добавлять
> агенту workflow-инструкции и ресурсы.
>
> В этой точке harness умеет искать `SKILL.md`, показывать каталог навыков
> модели, активировать skill через `activate_skill` и добавлять инструкции
> навыка в durable context.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Предыдущая ветка: `03-Context-Layer`
> Следующая ветка: `05-mcp`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Он даёт модели понятный цикл:
получить задачу, увидеть контекст проекта, выбрать подходящий skill, вызвать
инструменты и записать ход выполнения в trace.

Проект написан для Python 3.13+ и не имеет runtime-зависимостей. Внутри
используется OpenAI-совместимый API `/chat/completions`.

## Что есть в этой ветке

- команды `init`, `ask`, `run`, `trace` и `skills`;
- проектные инструкции `AGENTS.md`;
- слой контекста с бюджетом и `context_report`;
- инструмент `read_image` для моделей с vision input;
- базовые инструменты workspace: `list_files`, `read_file`, `write_file`,
  `search_code`, `apply_patch`, `run_shell`;
- discovery project-local skills в `.madharness_mini/skills` и `.agents/skills`;
- явная активация через `@skill:name`, `@skill/name`, `$name` и похожие фразы;
- auto-activation через catalog и инструмент `activate_skill`;
- CLI-диагностика `skills list`, `skills show`, `skills validate`.

В этой ветке ещё нет MCP, субагентов и hooks. Здесь фокус только на skills как
локальном расширении контекста и workflow-памяти агента.

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

Перейдите в корень продукта и создайте локальную настройку:

```bash
madharness-mini init \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --api-key "ключ-доступа-openrouter"
```

## Минимальный skill

Создайте файл `.madharness_mini/skills/docs-writer/SKILL.md`:

```md
---
name: docs-writer
description: Помогает обновлять README и учебную документацию проекта.
---

Перед правкой документации прочитай README, docs/README.md и связанные файлы.
Сохраняй короткий учебный стиль и не добавляй возможности, которых нет в коде.
```

Проверьте, что harness видит skill:

```bash
madharness-mini skills list
```

Запустите задачу с явным skill:

```bash
madharness-mini run "@skill:docs-writer обнови README"
```

Если skill не выбран явно, в `run` модель увидит компактный catalog и сможет
сама вызвать `activate_skill`.

## Документация ветки

- [Возможности ветки](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Слой контекста](docs/context-layer.md)
- [Agent Skills](docs/agent-skills.md)
- [Инструмент apply_patch](docs/apply-patch.md)

## Разработка самого проекта

Если вы меняете код `madharness-mini`, запускайте проверки из корня этого
репозитория:

```bash
uv run -m unittest discover -s tests
```

Быстрая ручная проверка CLI:

```bash
uv run madharness-mini skills validate
```

## Что дальше

Следующая ветка `05-mcp` добавляет подключение внешних инструментов через
минимальный stdio MCP-клиент.
