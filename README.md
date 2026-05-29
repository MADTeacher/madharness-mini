# madharness-mini

> Учебная ветка: `02-AGENTS-md`
>
> Тема главы: проектные инструкции `AGENTS.md` и первый мультимодальный
> инструмент.
>
> Эта ветка продолжает минимальный harness: агентский цикл уже умеет читать
> правила проекта из `AGENTS.md`, а инструмент `read_image` может передать
> модели локальное изображение, если выбранная модель поддерживает vision input.
>
> Лабораторные работы: [LABS.md](LABS.md)
> Предыдущая ветка: `01-minimalistic-harness`
> Следующая ветка: `03-Context-Layer`

`madharness-mini` — учебный минималистичный харнесс для работы кодирующего
ИИ-агента с локальным программным продуктом. Он даёт модели понятный цикл:
получить задачу, увидеть инструкции проекта, вызвать инструмент, получить
observation и продолжить работу.

Проект написан для Python 3.13+ и не имеет runtime-зависимостей. Внутри
используется OpenAI-совместимый API `/chat/completions`, поэтому можно
подключить OpenRouter, KodikRouter, локальный совместимый сервер или другой
сервис с тем же форматом API.

## Что есть в этой ветке

- команды `init`, `ask`, `run` и `trace`;
- базовые инструменты workspace: `list_files`, `read_file`, `write_file`,
  `search_code`, `apply_patch`, `run_shell`;
- инструмент `read_image` для PNG, JPEG, WEBP и неанимированного GIF;
- загрузка `AGENTS.md` из корня workspace и вложенных папок до текущей cwd;
- общий лимит на склейку проектных инструкций;
- политика безопасности для путей, защищённых файлов и shell-команд;
- JSONL-трассы запусков.

В этой ветке ещё нет отдельного слоя контекста, Agent Skills, MCP, субагентов и
hooks. Здесь проектные инструкции пока добавляются прямо в system prompt.

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
`protected_paths`, `allow_shell`, `supports_image_input`, `max_image_bytes` и
`image_detail`.

## Проектные инструкции

Добавьте в корень продукта файл `AGENTS.md`, если агент должен учитывать
локальные правила:

```md
# AGENTS.md

- Не добавляй зависимости без отдельного запроса.
- Для проверки запускай `uv run -m unittest discover -s tests`.
- Комментарии в коде пиши на русском.
```

Если команда запущена из вложенной папки, harness читает `AGENTS.md` по цепочке
от корня workspace до текущей директории. Общие правила идут раньше локальных.

## Первые команды

Задать вопрос без доступа к инструментам:

```bash
madharness-mini ask "Объясни, что делает этот проект"
```

Запустить агентский режим:

```bash
madharness-mini run "Найди команду для запуска тестов и объясни, что она проверяет"
```

Попросить модель учесть изображение:

```bash
madharness-mini run "Посмотри на 16.png и опиши, что важно проверить в коде"
```

Для работы `read_image` включите vision-ввод в конфиге:

```json
{
  "supports_image_input": true
}
```

Посмотреть краткую сводку трассы:

```bash
madharness-mini trace 20260529-171000
```

## Документация ветки

- [Возможности ветки](docs/capabilities.md)
- [Структура кода](docs/code-overview.md)
- [Проектные инструкции AGENTS.md](docs/project-instructions.md)

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

Следующая ветка `03-Context-Layer` отделяет сборку сообщений и бюджет контекста
от основного model/tool loop.
