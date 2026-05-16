# madharness-mini

`madharness-mini` - минималестичный харнесс для учебного проекта. Реализован на Python 3.13+ и не добавляет runtime-зависимостей сверх стандартной библиотеки.

## Установка через uv

Из корня проекта харнесс можно установить как системную команду через `uv tool install`:

```bash
uv tool install --python 3.13 .
```

После установки команда `madharness-mini` будет доступна из любой папки, если каталог инструментов `uv` добавлен в `PATH`.

Проверка:

```bash
madharness-mini ask "Return a short greeting"
```

Если команда не находится, выполните:

```bash
uv tool update-shell
```

Затем перезапустите терминал.

Если нужно установить харнесс из другой папки, передайте абсолютный путь к проекту:

```bash
uv tool install --python 3.13 /Users/madteacher/Documents/GitHub/mini-madharness
```

После изменений в коде переустановите команду:

```bash
uv tool install --python 3.13 --reinstall .
```

## Локальный запуск

Для разработки используйте `uv run`:

```bash
uv run madharness-mini ask "Return a short greeting"
```

## Инициализация проекта

В новом проекте сначала создайте локальную настройку харнесса:

```bash
uv run madharness-mini init --api-key "ваш-ключ-api"
```

Команда создаст `.madharness-mini/config.json` и каталог для трасс. Если ключ уже задан через переменную `MADHARNESS_MINI_API_KEY`, можно запустить:

```bash
uv run madharness-mini init
```

После этого `ask` и `run` будут брать настройки из `config.json`.

Тесты:

```bash
uv run -m unittest discover -s tests
```
