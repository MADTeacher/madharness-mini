# Agent Skills в madharness-mini

Agent Skill в `madharness-mini` — это проектная папка с инструкцией `SKILL.md` и дополнительными файлами, которые помогают агенту выполнить конкретный тип задач: писать документацию, проверять Python-проект, готовить релизные заметки, работать с диаграммами и т.п.

Навык не является плагином и не получает отдельного канала исполнения. Он добавляет агенту контекст и, при необходимости, указывает на bundled resources: scripts, references, assets и другие файлы внутри папки навыка. Все реальные действия всё равно проходят через обычные инструменты харнесса и общую политику безопасности.

## Зачем нужны навыки

Обычные проектные инструкции `AGENTS.md` подходят для устойчивых правил всего проекта: стиль кода, тестовая команда, ограничения безопасности. Agent Skills решают другую задачу: они описывают специализированный workflow, который нужен не всегда.

Например:

- `docs-writer` — как обновлять README и техническую документацию.
- `python-check` — как запускать локальную проверку проекта.
- `release-notes` — как собирать короткие релизные заметки.
- `diagramming` — как готовить редактируемые диаграммы и экспорт.

Главный принцип — прогрессивное раскрытие. Модель сначала видит только каталог навыков, а полный `SKILL.md` попадает в контекст только после активации конкретного навыка.

## Где лежат навыки

`madharness-mini` ищет навыки только внутри `workspace_root`, без отдельной настройки в `config.json`.

Поддерживаются два каталога:

```text
.madharness_mini/skills
.agents/skills
```

Каждая прямая подпапка с валидным `SKILL.md` считается кандидатом:

```text
.madharness_mini/
  skills/
    docs-writer/
      SKILL.md
      references/
      scripts/
      assets/

.agents/
  skills/
    shared-docs/
      SKILL.md
```

Если одно и то же `name` найдено в обоих каталогах, версия из `.madharness_mini/skills` перекрывает версию из `.agents/skills`. Это позволяет проекту переопределить общий interoperable skill локальной версией.

## Формат SKILL.md

`SKILL.md` состоит из YAML frontmatter и markdown-инструкций.

Минимальный пример:

```markdown
---
name: docs-writer
description: Используй этот навык для задач про README, docs и описание API.
---

# Docs Writer

Пиши коротко, проверяемо и со ссылками на конкретные файлы.
```

Обязательные поля:

| Поле | Смысл |
| --- | --- |
| `name` | Имя навыка. Используется в каталоге, CLI и `activate_skill`. |
| `description` | Короткое описание, по которому модель решает, нужен ли навык. |

Дополнительные поля:

| Поле | Смысл |
| --- | --- |
| `license` | Показывается в `skills show` и activation wrapper. |
| `compatibility` | Попадает в activation wrapper, чтобы модель видела требования среды. |
| `metadata` | Простая карта `ключ: значение` для диагностики и описания навыка. |
| `allowed-tools` | Experimental-подсказка для модели. Не отменяет глобальную политику безопасности. |

Тело markdown должно содержать основной workflow. Большие справочники, шаблоны, скрипты и ассеты лучше держать рядом в `references/`, `scripts/`, `assets/` и читать только тогда, когда workflow этого требует.

## Как работает run

Навыки подключаются только в режиме `run`. Режим `ask` остаётся коротким одиночным запросом: он не загружает skill catalog, не добавляет `activate_skill` и не разбирает skill-маркеры.

Поток `run` выглядит так:

1. `Config` определяет `workspace_root`.
2. Loader сканирует `.madharness_mini/skills` и `.agents/skills`.
3. Frontmatter каждого `SKILL.md` превращается в индекс навыков.
4. В трассу пишется `skills_discovered`.
5. Если пользователь явно указал skill, он активируется до первого обращения к модели.
6. Если явного выбора нет, модель получает компактный каталог навыков.
7. Когда модель решает, что skill нужен, она вызывает `activate_skill`.
8. Харнесс добавляет полный workflow навыка в durable context-фрагмент.
9. Модель продолжает работу уже с активными инструкциями.

## Каталог навыков

Каталог — это короткий системный фрагмент, который содержит только:

- `name`;
- `description`;
- путь к `SKILL.md` внутри workspace.

Полное тело `SKILL.md` на этом этапе не раскрывается. Это важно для бюджета контекста: проект может иметь несколько навыков, но модель не должна заранее получать все инструкции, справочники и шаблоны.

Пример смысла catalog-фрагмента:

```text
# Available Agent Skills

- `docs-writer`: Используй этот навык для задач про README...
  (location: `.agents/skills/docs-writer/SKILL.md`)
```

## Активация навыка

Есть два способа активации.

### Автоматический выбор агентом

Если пользователь не указал skill явно, в набор инструментов добавляется `activate_skill`. Его аргумент `name` ограничен enum найденных имён, поэтому модель может выбрать только реально доступный skill.

После вызова:

- observation сообщает, что skill активирован;
- в durable context добавляется полный workflow из `SKILL.md`;
- возвращается `skill_root`;
- перечисляются bundled resources без чтения их содержимого;
- в трассу пишется `skill_activated`.

### Явный выбор пользователем

Пользователь может указать skill в задаче:

```bash
madharness-mini run "@skill:docs-writer обнови README"
```

Поддерживаются формы:

- `@skill:docs-writer`;
- `@skill/docs-writer`;
- `$docs-writer`;
- фразы вида `используй навык docs-writer`.

Если skill найден, он активируется до первого model call. Если skill неизвестен, запуск завершается до обращения к модели:

```text
error: unknown skill: docs-writer
```

При явном выборе авто-подбор на этот запрос отключается: каталог не добавляется, а инструмент `activate_skill` не выдаётся модели. Пользовательский выбор считается более сильным сигналом, чем рассуждение модели.

## Durable context

Активированный skill добавляется как закреплённый `ContextFragment` с id вида:

```text
skill:<name>
```

Этот фрагмент защищён от удаления старой истории при бюджетной обрезке. Если контекст становится слишком большим, `ContextManager` может укорачивать tool outputs и удалять старые assistant/tool turns, но активные инструкции навыка остаются рядом с системным промптом.

Повторная активация того же skill не дублирует инструкции. Runtime помнит активные имена и возвращает observation `skill already active`.

## Bundled resources

Файлы внутри skill root считаются bundled resources. Обычно это:

```text
references/
scripts/
assets/
```

При активации харнесс перечисляет ресурсы:

- относительный путь внутри skill root;
- workspace-relative путь;
- тип по первой папке (`references`, `scripts`, `assets` или другое);
- размер файла в байтах.

Содержимое ресурсов не читается автоматически. Если workflow говорит прочитать справочник, модель должна вызвать обычный `read_file`:

```text
.agents/skills/docs-writer/references/style.md
```

Если workflow говорит запустить documented script, модель использует `run_shell` с `cwd`, равным skill root:

```json
{
  "command": "python scripts/check_project.py",
  "cwd": ".agents/skills/python-check"
}
```

Команда всё равно проходит обычную shell-политику: запрещены управляющие операторы, рискованные команды и выход за workspace.

## Безопасность

Skills не расширяют границы безопасности харнесса.

Основные правила:

- skill-каталоги должны находиться внутри `workspace_root`;
- `SKILL.md` читается как UTF-8 текст;
- symlink escape из skill root не считается bundled resource;
- scripts не запускаются сами по себе;
- resources читаются только обычными файловыми инструментами;
- `allowed-tools` не даёт разрешений сверх глобальной политики;
- `run_shell` по-прежнему проверяет команду через `Policy.shell_allowed`;
- файловые пути по-прежнему проверяются через `Policy.safe_path`.

Это значит, что skill может описать workflow, но не может обойти protected paths, выполнить скрытый shell-код или прочитать файл вне workspace.

## CLI-команды

Посмотреть найденные навыки:

```bash
madharness-mini skills list
```

Показать полный skill:

```bash
madharness-mini skills show docs-writer
```

Проверить диагностику discovery:

```bash
madharness-mini skills validate
```

`validate` полезен перед запуском агента: он покажет ошибки frontmatter, пустой `description`, слишком большой `SKILL.md`, коллизии имён и предупреждения вроде несовпадения `name` с именем папки.

## Трассы

Каждый `run` пишет JSONL-трассу в:

```text
.madharness-mini/traces/*.jsonl
```

Для skills есть отдельные события:

| Событие | Что означает |
| --- | --- |
| `skills_discovered` | Харнесс просканировал skill-каталоги и записал имена плюс диагностику. |
| `skill_activated` | Skill стал active context-фрагментом. |
| `skill_resource_used` | Инструмент обратился к файлу или cwd внутри активного skill root. |

Команда `trace` показывает короткую сводку:

```bash
madharness-mini trace 20260528-104532
```

Пример строки:

```text
skills: discovered 2; activated lesson-docs; resources used 1
```

Полный текст активированного `SKILL.md` не дублируется в трассе. В `context_report` видны только id, source и размер фрагмента. Если модель сама прочитала bundled reference через `read_file`, содержимое этого файла попадёт в tool observation — это ожидаемо, потому что ресурс был реально использован.

## Типичный сценарий

1. Пользователь добавляет skill:

```text
.agents/skills/docs-writer/SKILL.md
```

2. Проверяет discovery:

```bash
madharness-mini skills validate
```

3. Запускает задачу обычным языком:

```bash
madharness-mini run "Обнови README и проверь стиль документации"
```

4. Модель видит catalog и вызывает `activate_skill` для `docs-writer`.
5. Харнесс добавляет workflow skill в durable context.
6. Модель читает нужные файлы и bundled references.
7. Модель вносит изменения через обычные инструменты.
8. Пользователь смотрит `trace`, чтобы убедиться, что skill был активирован.

## Как писать хороший skill

Хороший `SKILL.md` отвечает на три вопроса:

- когда использовать навык;
- что агент должен сделать по шагам;
- какие bundled resources нужны и когда их читать или запускать.

Практические правила:

- `description` должен быть достаточно конкретным, чтобы модель могла выбрать skill по задаче.
- Основной workflow держите в `SKILL.md`, а большие материалы выносите в `references/`.
- Скрипты кладите в `scripts/` и явно пишите команду запуска.
- Не обещайте доступ к инструментам, которые запрещены глобальной политикой.
- Не храните секреты в skill resources.
- Проверяйте навык через `skills validate` и один маленький smoke-run.

## Частые проблемы

Skill не найден:

- проверьте, что путь ровно `.madharness_mini/skills/<name>/SKILL.md` или `.agents/skills/<name>/SKILL.md`;
- проверьте `name` во frontmatter;
- запустите `madharness-mini skills list`.

Skill найден, но модель не активирует его:

- уточните `description`;
- прямо напишите в задаче, что нужен workflow этого типа;
- используйте явный маркер `@skill:<name>`.

Скрипт не запускается:

- проверьте, что команда не содержит `|`, `>`, `<`, `&&`, `||`, `;`;
- укажите `cwd` равным skill root;
- убедитесь, что скрипт не пытается выйти за workspace.

Трасса не показывает текст `SKILL.md`:

- это нормальное поведение;
- смотрите `skill_activated` и `context_report.fragments`;
- полный текст доступен через `madharness-mini skills show <name>`.

## Связанные файлы реализации

Основные модули:

- `madharness_mini/skills/loader.py` — discovery и frontmatter parser.
- `madharness_mini/skills/catalog.py` — catalog и явный выбор.
- `madharness_mini/skills/activation.py` — activation wrapper, resources и runtime active skills.
- `madharness_mini/skills/provider.py` — `ContextProvider` catalog и `ToolProvider` для `activate_skill`.
- `madharness_mini/loop.py` — подключение skills к `run`.
- `madharness_mini/tools/shell.py` — `cwd` для documented skill scripts.
- `madharness_mini/trace.py` — summary skill events.
