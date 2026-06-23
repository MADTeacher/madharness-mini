"""Чистка сообщений модели под хранение в истории.

Функции отсюды нормализуют assistant-ответ и tool-наблюдение так, чтобы их
можно было безопасно положить в историю и переотправить в следующем Chat
request: убирают лишние поля, чинят битый JSON arguments, клипают избыточный
текст и сворачивают тяжёлые write/patch-args в дайджест. Слой не имеет
состояния и не зависит от ContextManager — только чистые преобразования
dict'ов.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .budget import clip_text, digest_write_args
from ..utils import paths_from_patch

# Полный assistant-текст полезен только до разумного предела: модель уже
# получила свои рассуждения в прошлом ходе, а следующий запрос платит за них снова.
ASSISTANT_CONTENT_LIMIT = 8000

# Если assistant сразу вызывает tool, его текст обычно служебный; сохраняем кратко.
ASSISTANT_TOOL_CONTENT_LIMIT = 2000

# Аргументы write_file/apply_patch старого хода сворачиваем в указатель, если они
# крупнее этого порога: тело файла (код) доминирует в стоимости assistant-ходов.
SUMMARY_TOOLCALL_LIMIT = 200


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Сохраняем в историю только поля, которые нужны следующему Chat request."""

    tool_calls = _sanitize_tool_calls(message.get("tool_calls") or [])
    content = message.get("content")
    limit = ASSISTANT_TOOL_CONTENT_LIMIT if tool_calls else ASSISTANT_CONTENT_LIMIT
    stored: dict[str, Any] = {"role": "assistant"}
    if isinstance(content, str):
        stored["content"] = clip_text(content, limit)
    elif content is None:
        stored["content"] = None if tool_calls else ""
    else:
        stored["content"] = copy.deepcopy(content)
    if tool_calls:
        stored["tool_calls"] = tool_calls
    return stored


def _sanitize_tool_calls(calls: list[Any]) -> list[dict[str, Any]]:
    """Оставляем у tool call только OpenAI-compatible id, type и function."""

    sanitized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        # Гарантируем, что в историю не попадёт обрезанный/битый JSON: если модель
        # упёрлась в лимит токенов внутри строкового литерала, arguments останется
        # незакрытым, и провайдер отвергнет следующий запрос с HTTP 400. Невалидную
        # строку заменяем на '{}' и сохраняем обрезок для диагностики в trace.
        sanitized_call: dict[str, Any] = {
            "id": str(call.get("id") or name),
            "type": str(call.get("type") or "function"),
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
        try:
            json.loads(arguments)
        except (ValueError, TypeError):
            sanitized_call["function"]["arguments"] = "{}"
            sanitized_call["_malformed_arguments"] = arguments[:500]
        sanitized.append(sanitized_call)
    return sanitized


def _digest_old_write_tool_calls(message: dict[str, Any]) -> set[str]:
    """Сворачиваем аргументы write_file/apply_patch у старого assistant-хода.

    Заменяем тело файла (content/patch) дайджестом-указателем, оставляя валидный
    JSON: иначе провайдер отвергнет следующий запрос. Возвращаем множество путей,
    чьи write/patch-args действительно свернулись (длиннее SUMMARY_TOOLCALL_LIMIT),
    чтобы вышележащий _summarize_old_entries пометил их write-collapsed: текст последней
    правки пути покинул промпт, модель может действовать вслепую по нему.

    Пустое множество = ход не изменён (прежний bool(False)). Это сохраняет
    прежнюю семантику «changed» в вызывающей стороне через bool(digested_paths).
    """

    digested_paths: set[str] = set()
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if name not in ("write_file", "apply_patch"):
            continue
        if not isinstance(arguments, str) or len(arguments) <= SUMMARY_TOOLCALL_LIMIT:
            continue
        # Достаём пути до замены arguments на digest — из исходных args надёжнее,
        # чем из digest-вывода (хотя digest_write_args тоже сохраняет path/paths).
        paths = _paths_from_write_args(name, arguments)
        digested = digest_write_args(arguments, name)
        if digested != arguments:
            function["arguments"] = digested
            digested_paths.update(paths)
    return digested_paths


def _paths_from_write_args(name: str, arguments: str) -> list[str]:
    """Пути, затронутые write_file/apply_patch tool_call, из JSON-arguments.

    write_file — один путь под ключом path; apply_patch — мультфильмный, пути
    достаём общим парсером patch-формата. Любая ошибка разбора возвращает пустой
    список: лучше потерять collapsed-пометку, чем уронить свёртку ходов.
    """

    try:
        payload = json.loads(arguments)
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    if name == "write_file":
        path = payload.get("path")
        return [str(path)] if isinstance(path, str) and path else []
    if name == "apply_patch":
        patch = payload.get("patch")
        if isinstance(patch, str):
            return paths_from_patch(patch)
    return []


def _tool_kind_and_path(content: str) -> tuple[str, str | None]:
    """Достаём имя инструмента и путь из JSON-сериализованного tool-наблюдения.

    Нужно, чтобы возрастная эвикция различала read_file (сворачиваем в указатель)
    и прочие tool outputs (обрезаем). Путь может отсутствовать в observation —
    тогда вызывающая сторона подставит его из file_refs.
    """

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "", None
    if not isinstance(payload, dict):
        return "", None
    tool_name = str(payload.get("tool") or "")
    path_value = str(payload.get("path") or "") or None
    return tool_name, path_value


def _tool_name(call: dict[str, Any], observation: dict[str, Any]) -> str:
    """Достаём имя инструмента для fallback tool_call_id."""

    function = call.get("function") or {}
    return str(function.get("name") or observation.get("tool") or "tool_call")
