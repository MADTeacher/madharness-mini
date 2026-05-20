"""Константы и вспомогательные функции, которые используются в разных модулях харнесса.

Здесь собраны настройки по умолчанию, ограничение длины ответов инструментов,
единый формат результатов для модели и заготовки для описания параметров инструментов.
"""

import json
from pathlib import Path
from typing import Any

# Каталог внутри проекта, куда харнесс складывает служебные файлы (трассы, состояние).
STATE_DIR = ".madharness-mini"

# Максимальная длина текста, который один вызов инструмента может вернуть модели.
MAX_OUTPUT = 20000

# Значения конфигурации по умолчанию, если пользователь ничего не задал в 
# файле .env или переменных окружения
DEFAULT_CONFIG = {
    "model": "deepseek/deepseek-v4-flash",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "",
    "temperature": 0.2,
    "max_turns": 50,
    "workspace_root": ".",
    "protected_paths": [".git", ".env", "secrets", "~/.ssh"],
    "allow_shell": True,
}


def clipped(text: str, limit: int = MAX_OUTPUT) -> str:
    """Укорачиваем слишком длинный вывод инструмента перед отправкой модели.

    Если команда или чтение файла вернули огромный текст, модель получит только
    начало и пометку, сколько символов обрезано — иначе контекст переполнится.
    """

    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped {len(text) - limit} chars]"


def ok(
    tool: str,
    summary: str,
    **data: Any,
) -> dict[str, Any]:
    """Сообщаем модели, что инструмент отработал успешно.

    Возвращаем словарь с полями ok, tool, summary и при необходимости дополнительными
    данными (содержимое файла, stdout команды и т.д.). Модель читает это как результат
    вызова инструмента в следующем шаге диалога.
    """

    return {"ok": True, "tool": tool, "summary": summary, **data}


def fail(
    tool: str,
    summary: str,
    **data: Any,
) -> dict[str, Any]:
    """Сообщаем модели, что инструмент не смог выполнить задачу.

    Тот же формат, что у ok, но ok=False: отказ политики безопасности, неверный путь,
    ошибка shell и т.п. Краткий summary объясняет причину человекочитаемым текстом.
    """

    return {"ok": False, "tool": tool, "summary": summary, **data}


def ignored(path: Path) -> bool:
    """Решаем, пропускать ли этот путь при поиске и листинге файлов в workspace.

    Служебные каталоги (.git, кэш Python, виртуальное окружение, каталог харнесса)
    агенту обычно не нужны; их не показываем в glob/grep/list.
    """

    ignored_names = {".git", STATE_DIR, "__pycache__", ".venv", ".uv-cache"}
    return any(part in ignored_names for part in path.parts)


def parse_tool_args(
    call: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Достаём из ответа модели имя инструмента и его аргументы.

    Модель присылает tool_call с полем function: name и arguments (строка JSON
    или уже словарь). Приводим arguments к dict и возвращаем пару (имя, аргументы)
    для диспетчера инструментов в tools.py.
    """

    fn = call.get("function", {})
    raw = fn.get("arguments") or "{}"
    args = raw if isinstance(raw, dict) else json.loads(raw)
    return fn.get("name", ""), args


def obj(
    props: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Описываем набор полей аргументов инструмента как JSON Schema-объект.

    OpenAI-совместимый API ожидает у каждого инструмента parameters с type: object,
    списком properties и required. Эта функция собирает такой фрагмент схемы.
    """

    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


def strp(
    default: str | None = None,
    desc: str = "",
    req: bool = False,
) -> dict[str, Any]:
    """Описываем один строковый параметр в схеме аргументов инструмента.

    desc попадает в description для модели; default подставляется, если параметр
    необязательный (req=False).
    """

    data: dict[str, Any] = {"type": "string", "description": desc}
    if default is not None and not req:
        data["default"] = default
    return data


def intp(default: int) -> dict[str, Any]:
    """Описываем один целочисленный параметр со значением по умолчанию в схеме инструмента."""

    return {"type": "integer", "default": default}
