# madharness-mini

> Учебная ветка: `07-hooks`
>
> Тема главы: lifecycle hooks как проектный слой аудита и блокировки действий
> агента.
>
> В этой точке harness умеет читать `.madharness-mini/hooks.json`, отправлять
> JSON-события локальным command hooks, писать hook-аудит в trace и блокировать
> `before_tool_call` до запуска handler-а.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Предыдущая ветка: `06-subagents`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Финальная учебная ветка показывает
полный набор механизмов: workspace tools, проектные инструкции, context layer,
Agent Skills, MCP, субагентов и hooks.

Проект написан для Python 3.13+ и не имеет runtime-зависимостей. Внутри
используется OpenAI-совместимый API `/chat/completions`.

## Что есть в этой ветке

- команды `init`, `ask`, `run`, `trace`, `skills` и `subagents`;
- проектные инструкции `AGENTS.md`;
- слой контекста с бюджетом и `context_report`;
- project-local Agent Skills и `activate_skill`;
- stdio MCP tools через `.madharness-mini/mcp.json`;
- markdown-субагенты, `delegate_task`, `ask_user` и дочерние traces;
- lifecycle hooks из `.madharness-mini/hooks.json`;
- redaction payload перед передачей hook-команде;
- блокировка tool call через `before_tool_call`.

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

Запустите агентский режим:

```bash
madharness-mini run "Найди команду для запуска тестов и объясни, что она проверяет"
```

## Минимальный hook

Создайте `.madharness-mini/hooks.json`:

```json
{
  "hooks": [
    {
      "id": "deny-shell",
      "event": "before_tool_call",
      "match": { "tool": "run_shell" },
      "command": "python3",
      "args": ["scripts/hooks/deny_shell.py"],
      "cwd": ".",
      "timeout_seconds": 3
    }
  ]
}
```

Hook-команда получает событие в stdin. Если она печатает:

```json
{ "ok": false, "block": "shell запрещён правилами проекта" }
```

handler инструмента не запускается, а модель получает обычное fail-observation.
Остальные события нужны для аудита и диагностики; сейчас блокировать действие
может только `before_tool_call`.

## Документация ветки

- [Возможности ветки](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Слой контекста](docs/context-layer.md)
- [Agent Skills](docs/agent-skills.md)
- [MCP](docs/mcp.md)
- [Субагенты](docs/subagents.md)
- [Hooks](docs/hooks.md)
- [Инструмент apply_patch](docs/apply-patch.md)

## Разработка самого проекта

Если вы меняете код `madharness-mini`, запускайте проверки из корня этого
репозитория:

```bash
uv run -m unittest discover -s tests
```

Быстрая ручная проверка CLI:

```bash
uv run madharness-mini run "Объясни, какие hooks подключены в этом проекте"
```

## Как читать эту ветку

Это самая полная учебная точка. Если вы пришли из книги или курса, сначала
посмотрите [LABS.md](LABS.md), затем откройте [docs/README.md](docs/README.md)
и переходите в конкретный документ по механизму, который сейчас изучаете.
