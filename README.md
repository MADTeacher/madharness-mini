# madharness-mini

`madharness-mini` - это учебный харнесс для работы с ИИ-агентом над разрабатываемым приложением. Проект написан на Python 3.13+, без использования дополнительных пакетов.

## Быстрый запуск

Установите `madharness-mini` из GitHub:

```bash
uv tool install madharness-mini --from git+https://github.com/MADTeacher/madharness-mini.git
```

Откройте папку проекта, с которой должен работать агент, и создайте настройку:

```bash
madharness-mini init \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --api-key "ключ-доступа-openrouter"
```

Команда создаёт файл `.madharness-mini/config.json`. После инициализации в нём можно поменять настройки проекта: `model`, `base_url`, `api_key`, `temperature`, `max_turns`, `workspace_root`, `protected_paths` и `allow_shell`. Подробнее поля описаны в разделе [Возможности харнесса](docs/capabilities.md#настройки).

Задайте простой вопрос:

```bash
madharness-mini ask "Объясни, что делает этот проект"
```

Запустите агентский режим для задачи по проекту:

```bash
madharness-mini run "Найди команду для запуска тестов и объясни, что она проверяет"
```

При необходимости добавьте проектный skill в `.madharness_mini/skills/<name>/SKILL.md` или `.agents/skills/<name>/SKILL.md`. В режиме `run` его можно подключить явно:

```bash
madharness-mini run "@skill:docs-writer обнови README"
```

Для подключения stdio MCP-сервера добавьте отдельный файл `.madharness-mini/mcp.json`. Например, для Playwright MCP:

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

Если терминал не находит команду `madharness-mini`, выполните:

```bash
uv tool update-shell
```

Потом закройте терминал и откройте его заново.

## Дополнительная документация

- [Возможности харнесса](docs/capabilities.md): режимы, инструменты агента, настройки и ограничения безопасности.
- [Структура кода](docs/code-overview.md): модули проекта и поток выполнения агентского режима.
- [Слой контекста](docs/context-layer.md): как собираются сообщения для модели и как расширения добавляют свой контекст.
- [Agent Skills](docs/agent-skills.md): принцип работы проектных навыков, активация, ресурсы, безопасность и трассы.
- [План поддержки Agent Skills](docs/agent-skills-plan.html): формат навыков, подключение через контекст и этапы внедрения.
- [MCP](docs/mcp.md): подключение stdio MCP-серверов, формат `mcp.json`, lifecycle, безопасность и трассы.

## Разработка самого проекта

Если вы меняете код `madharness-mini`, запускайте команду из корня этого репозитория через `uv run`, предварительно настроив .env:

```bash
uv run madharness-mini ask "Объясни, что делает этот проект"
```

Переустановить харнесс из локального проекта можно с помощью команды:

```bash
uv tool install --python 3.13 --force .
```

## Проверка

Для запуска тестов введите следующую команду:

```bash
uv run -m unittest discover -s tests
```
