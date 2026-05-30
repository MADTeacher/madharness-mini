# madharness-mini

> Учебная ветка: `05-mcp`
>
> Тема главы: подключение внешних инструментов через минимальный stdio MCP
> client без runtime-зависимостей.
>
> В этой точке harness умеет читать `.madharness-mini/mcp.json`, запускать
> явно включённые MCP-серверы, получать `tools/list` и отдавать модели MCP tools
> как обычные `ToolSpec` с префиксом `mcp__server__tool`.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Предыдущая ветка: `04-Agents-Skills`
> Следующая ветка: `06-subagents`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Он показывает, как локальный
agent loop можно расширить внешними инструментами, не превращая учебный проект
в тяжёлый SDK.

Проект написан для Python 3.13+ и не имеет runtime-зависимостей. Внутри
используется OpenAI-совместимый API `/chat/completions`.

## Что есть в этой ветке

- команды `init`, `ask`, `run`, `trace` и `skills`;
- проектные инструкции `AGENTS.md`;
- слой контекста с бюджетом и `context_report`;
- project-local Agent Skills и `activate_skill`;
- инструмент `read_image` для моделей с vision input;
- базовые инструменты workspace: `list_files`, `read_file`, `write_file`,
  `search_code`, `apply_patch`, `run_shell`;
- пакет `madharness_mini.mcp` со stdio transport, JSON-RPC и provider-ом tools;
- отдельный файл `.madharness-mini/mcp.json` для MCP-серверов;
- нормализация MCP `tools/call` в обычное observation harness.

В этой ветке ещё нет субагентов и hooks. MCP здесь изучается как отдельный
источник инструментов, а не как система оркестрации ролей.

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

Если терминал после установки не видит команду:

```bash
uv tool update-shell
```

## Подключение MCP-сервера

Создайте `.madharness-mini/mcp.json`:

```json
{
  "servers": {
    "playwright": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"],
      "cwd": ".",
      "timeout_seconds": 30
    }
  }
}
```

Если файла нет, MCP выключен. Harness запускает только явно включённые серверы.
Команда и аргументы передаются в `subprocess` списком, без shell-строки.

После `initialize` harness вызывает `tools/list`. Например MCP tool
`browser_navigate` сервера `playwright` станет tool name:

```text
mcp__playwright__browser_navigate
```

Для модели это обычный инструмент с JSON Schema, а результат `tools/call`
приводится к обычному observation.

## Документация ветки

- [Возможности ветки](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Слой контекста](docs/context-layer.md)
- [Agent Skills](docs/agent-skills.md)
- [MCP](docs/mcp.md)
- [Инструмент apply_patch](docs/apply-patch.md)

## Разработка самого проекта

Если вы меняете код `madharness-mini`, запускайте проверки из корня этого
репозитория:

```bash
uv run -m unittest discover -s tests
```

Быстрая ручная проверка CLI:

```bash
uv run madharness-mini run "Найди доступные инструменты и объясни, что они делают"
```

## Что дальше

Следующая ветка `06-subagents` добавляет оркестрацию ролей: parent agent сможет
делегировать задачи встроенным markdown-субагентам.

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
