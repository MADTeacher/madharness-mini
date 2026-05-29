# Возможности ветки

`04-Agents-Skills` добавляет к context layer project-local skills. Skill — это
папка с `SKILL.md`, который описывает workflow и ресурсы для конкретного типа
задач.

## Команды CLI

| Команда | Что делает |
| --- | --- |
| `madharness-mini init` | Создаёт или обновляет `.madharness-mini/config.json`. |
| `madharness-mini ask "..."` | Отправляет один запрос модели без tools и без skills catalog. |
| `madharness-mini run "..."` | Запускает agent loop с инструментами, контекстом и skills. |
| `madharness-mini trace <id>` | Показывает краткую сводку JSONL-трассы. |
| `madharness-mini skills list` | Показывает найденные project-local skills. |
| `madharness-mini skills show <name>` | Показывает metadata, ресурсы и текст одного skill. |
| `madharness-mini skills validate` | Печатает диагностику `SKILL.md`. |

## Где лежат skills

Discovery смотрит прямые подпапки:

```text
.madharness_mini/skills
.agents/skills
```

В каждой папке skill должен иметь `SKILL.md`. Обязательные поля frontmatter:
`name` и `description`. Поддерживаются optional-поля `license`,
`compatibility`, `metadata` и experimental `allowed-tools`.

## Как skill попадает в контекст

Есть два пути:

- пользователь явно выбирает skill через `@skill:name`, `@skill/name`, `$name`
  или фразу вида `используй навык name`;
- модель видит compact catalog и вызывает `activate_skill`.

После активации тело `SKILL.md` становится durable `ContextFragment`: оно
сохраняется при обрезке обычной истории и влияет на следующие model calls.

## Инструменты режима `run`

К инструментам предыдущей ветки добавляется `activate_skill`. Он появляется
только в `run`, когда найдены skills и пользователь не выбрал skill явно.

Базовые инструменты остаются прежними: `list_files`, `read_file`, `read_image`,
`search_code`, `apply_patch`, `write_file`, `run_shell`.

## Trace

Для skills важны события:

- `skills_discovered`;
- `skill_activated`;
- `skill_resource_used`.

Trace не дублирует полный текст `SKILL.md`. Это сохраняет журнал компактным и
не раскрывает лишний контент.

## Что не входит в эту ветку

Ветка не содержит MCP, субагентов и hooks. Skills не дают особых прав: чтение
ресурсов и запуск scripts всё равно идут через обычные инструменты и общую
политику workspace.
