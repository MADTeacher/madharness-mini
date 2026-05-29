# Возможности ветки

`06-subagents` добавляет оркестрацию markdown-субагентов. Parent agent может
делегировать подзадачу роли, которая получает отдельный prompt, allow-list tools,
лимиты и собственный trace.

## Команды CLI

| Команда | Что делает |
| --- | --- |
| `madharness-mini init` | Создаёт или обновляет `.madharness-mini/config.json`. |
| `madharness-mini ask "..."` | Отправляет один запрос модели без tools. |
| `madharness-mini run "..."` | Запускает parent agent loop. |
| `madharness-mini trace <id>` | Показывает краткую сводку JSONL-трассы. |
| `madharness-mini skills list/show/validate` | Диагностирует project-local Agent Skills. |
| `madharness-mini subagents list/show/validate` | Диагностирует встроенных и project-local субагентов. |

## Оркестрация

Режим задаётся полем `orchestration_mode` или CLI-флагами:

| Режим | Поведение |
| --- | --- |
| `off` | `delegate_task` не добавляется. |
| `requested` | Делегация появляется только при явном запросе пользователя. |
| `auto` | Parent видит `delegate_task`, но может решить задачу сам. |
| `required` | Parent выступает координатором и делегирует правки ролям. |

## Субагенты

Встроенные роли лежат в `madharness_mini/prompts/subagents/`: `researcher`,
`planner`, `implementer`, `reviewer`.

Project-local роли добавляются в:

```text
.madharness-mini/subagents/<name>.md
```

Frontmatter задаёт `name`, `description`, `profile`, `tools`, `max_turns` и
другие лимиты. Markdown-body становится системным prompt роли.

`profile` помогает описать тип роли, но фактические полномочия задаёт список
`tools`. Субагент не получает `delegate_task`, чтобы не запускать рекурсивную
оркестрацию.

## Инструменты

К tools предыдущей ветки добавляются:

| Инструмент | Где доступен |
| --- | --- |
| `delegate_task` | Parent agent, если режим оркестрации разрешает делегацию. |
| `ask_user` | Только внутри субагента, если указан в его `tools`. |

`ask_user` не читает stdin. Он завершает текущую делегацию статусом
`needs_user_input`, а основной `run` печатает вопрос пользователю.

## Trace

У субагента появляется отдельный дочерний trace-файл. Родительская трасса пишет
`subagent_started`, `subagent_finished` или `subagent_failed` и ссылку на этот
локальный trace.

## Что не входит в эту ветку

Здесь ещё нет hooks. Субагенты управляют ролями и делегацией, но не добавляют
lifecycle-политику перед каждым tool call.
