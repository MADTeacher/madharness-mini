"""Оценка токенового бюджета и обрезка больших tool-наблюдений."""

import hashlib
import json
from collections.abc import Callable
from typing import Any

from ..utils import paths_from_patch
from .fragments import ContextFragment
from .history import HistoryEntry

# Консервативная цена токена без tokenizer конкретной модели.
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 3


def estimate_tokens(payload: Any) -> int:
    """Оцениваем токены по размеру компактного JSON в UTF-8.

    Харнесс работает с OpenAI-совместимыми провайдерами, где tokenizer зависит
    от выбранной модели. Поэтому бюджет намеренно использует один простой
    приближённый счётчик без runtime-зависимостей.
    """

    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return (
        len(raw) + TOKEN_ESTIMATE_BYTES_PER_TOKEN - 1
    ) // TOKEN_ESTIMATE_BYTES_PER_TOKEN


def estimate_request_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Оцениваем части запроса, которые занимают контекст модели."""

    payload: dict[str, Any] = {"messages": messages}
    messages_tokens = estimate_tokens(messages)
    tools_tokens = 0
    if tools:
        payload["tools"] = tools
        tools_tokens = estimate_tokens(tools)
    return {
        "messages_tokens_estimate": messages_tokens,
        "tools_tokens_estimate": tools_tokens,
        "request_tokens_estimate": estimate_tokens(payload),
    }


def clip_tool_messages(entries: list[HistoryEntry], limit: int) -> list[dict[str, Any]]:
    """Укорачиваем content у role=tool сообщений и описываем обрезанные места."""

    clipped: list[dict[str, Any]] = []
    for entry in entries:
        for message in entry.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or len(content) <= limit:
                continue
            shortened = clip_tool_content(content, limit)
            message["content"] = shortened
            clipped.append(
                {
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "before_chars": len(content),
                    "after_chars": len(shortened),
                    "saved_chars": len(content) - len(shortened),
                }
            )
    return clipped


def clip_tool_content(content: str, limit: int) -> str:
    """Сохраняем краткое JSON-наблюдение, когда полный output слишком велик."""

    excerpt = clip_text(content, max(40, limit // 2))
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return clip_text(content, limit)
    if not isinstance(payload, dict):
        return clip_text(content, limit)
    compact = {key: payload[key] for key in ("ok", "tool", "summary") if key in payload}
    compact["_context_truncated"] = True
    compact["content_excerpt"] = excerpt
    rendered = json.dumps(compact, ensure_ascii=False)
    if len(rendered) <= limit:
        return rendered
    return clip_text(content, limit)


def clip_text(text: str, limit: int) -> str:
    """Обрезаем строку с явной пометкой для модели."""

    if len(text) <= limit:
        return text
    marker = f"\n...[context clipped {len(text) - limit} chars]"
    keep = max(limit - len(marker), 0)
    return text[:keep] + marker


def dedup_tool_messages(
    entries: list[HistoryEntry],
    fragments: list[ContextFragment],
    is_protected_read: Callable[[str, int], bool] | None = None,
) -> list[dict[str, Any]]:
    """Сворачиваем избыточные role=tool наблюдения в краткие сводки.

    Два правила, оба сохраняют ok/tool/summary, чтобы модель не потеряла факт
    успешного вызова:
      • path_match — read_file по пути, который уже лежит в окне как постоянный
        фрагмент (например AGENTS.md). Полный текст дублирует фрагмент, поэтому
        наблюдение заменяется пометкой "content already in context".
      • intra_history — повтор идентичного observation позже в истории. Оставляем
        самое свежее вхождение, более старые сворачиваем.

    Пути берём из entry.file_refs (надёжно), а не из парсинга summary. Возвращает
    описания свёрнутых сообщений для отчёта трассы, мутирует content in-place.

    is_protected_read — единый предикат защиты чтения (FL1). Если задан и вернул
    True для пары (path, turn) read_file-наблюдения, такое чтение не сворачивается:
    файл позже правился, и полный текст ещё нужен модели. При None поведение
    прежнее.
    """

    constant_sources = {
        fragment.source for fragment in fragments if not fragment.transient
    }
    deduped: list[dict[str, Any]] = []
    # Считаем хэши content role=tool по всей истории, чтобы найти повторяющиеся.
    content_hash_to_turns: dict[str, list[int]] = {}
    for turn, entry in enumerate(entries):
        for message in entry.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            content_hash_to_turns.setdefault(digest, []).append(turn)

    for turn, entry in enumerate(entries):
        # Карта path -> kind для этого хода из файловых ссылок.
        path_kinds = {ref.path: ref.kind for ref in entry.file_refs}
        for message in entry.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            rule = _dedup_rule_for(
                message, content, path_kinds, constant_sources, content_hash_to_turns, turn
            )
            if rule is None:
                continue
            before_chars = len(content)
            # Для read_file строим дайджест-указатель (путь + диапазон строк),
            # а не обрезок текста: модель сохраняет знание, что и где читала.
            read_path = _read_file_path(content, path_kinds, rule)
            # Защита актуального чтения: если это read_file по пути, который позже
            # правился, оставляем полное наблюдение — иначе модель будет править
            # файл по устаревшему дайджесту (cold_gap).
            if (
                is_protected_read is not None
                and read_path is not None
                and _is_read_file_observation(content)
                and is_protected_read(read_path, turn)
            ):
                continue
            shortened = _dedup_summary(message, content, rule, read_path)
            message["content"] = shortened
            deduped.append(
                {
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "rule": rule,
                    "turn": turn,
                    "before_chars": before_chars,
                    "after_chars": len(shortened),
                    "saved_chars": before_chars - len(shortened),
                }
            )
    return deduped


def digest_read_file(content: str, path: str | None) -> str:
    """Осмысленный дайджест старого read_file-наблюдения.

    Сохраняем указатель — путь и диапазон строк — но роняем сам текст. Модель
    видит, что и где она читала раньше, и может перечитать свежее состояние.
    Используется и в дедупе (D), и в возрастной эвикции (B): вместо слепой
    обрезки середины текста оставляем связный «скелет» наблюдения.
    """

    path_value = path
    lines_value: str | None = None
    ok_value = True
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            ok_value = bool(payload.get("ok", True))
            if not path_value:
                path_value = str(payload.get("path") or "") or None
            start = payload.get("start")
            end = payload.get("end")
            if isinstance(start, int) and isinstance(end, int):
                lines_value = f"{start}-{end}"
    except json.JSONDecodeError:
        pass
    compact: dict[str, Any] = {"ok": ok_value, "tool": "read_file"}
    if path_value:
        compact["path"] = path_value
    if lines_value:
        compact["lines"] = lines_value
    compact["_context_digested"] = True
    compact["note"] = "content read earlier; call read_file for the current state"
    return json.dumps(compact, ensure_ascii=False)


def digest_write_args(arguments: str, tool_name: str) -> str:
    """Осмысленный дайджест аргументов старого write_file/apply_patch.

    Тело файла (content у write_file, patch у apply_patch) — это обычно весь код,
    который модель уже применила к диску. В истории он переотправляется каждый ход
    и доминирует в стоимости assistant-сообщений (см. анализ тепловых карт). По
    аналогии с digest_read_file сохраняем указатель — путь(и) и размер payload —
    но роняем сам текст, оставляя валидный JSON и подсказку перечитать файл. При
    неразборном JSON или отсутствии тела возвращаем исходную строку: лучше оставить
    как есть, чем сломать запрос к провайдеру обрезанным JSON.
    """

    try:
        payload = json.loads(arguments)
    except (ValueError, TypeError):
        return arguments
    if not isinstance(payload, dict):
        return arguments
    if tool_name == "write_file":
        content = payload.get("content")
        if not isinstance(content, str):
            return arguments
        compact: dict[str, Any] = {"_context_digested": True, "tool": "write_file"}
        path = payload.get("path")
        if isinstance(path, str) and path:
            compact["path"] = path
        compact["content_chars"] = len(content)
        compact["note"] = "content written earlier; call read_file for the current state"
        return json.dumps(compact, ensure_ascii=False)
    if tool_name == "apply_patch":
        patch = payload.get("patch")
        if not isinstance(patch, str):
            return arguments
        compact = {"_context_digested": True, "tool": "apply_patch"}
        paths = paths_from_patch(patch)
        if paths:
            compact["paths"] = paths
        compact["patch_chars"] = len(patch)
        compact["note"] = "patch applied earlier; call read_file for the current state"
        return json.dumps(compact, ensure_ascii=False)
    return arguments


def _is_read_file_observation(content: str) -> bool:
    """Истинно, если JSON-наблюдение принадлежит инструменту read_file."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("tool") == "read_file"


def _read_file_path(content: str, path_kinds: dict[str, str], rule: str) -> str | None:
    """Достаём путь read_file-наблюдения: сначала из file_refs, потом из payload."""

    for path, kind in path_kinds.items():
        if kind == "read":
            return path
    try:
        payload = json.loads(content)
        if isinstance(payload, dict) and payload.get("tool") == "read_file":
            return str(payload.get("path") or "") or None
    except json.JSONDecodeError:
        pass
    return None


def _dedup_rule_for(
    message: dict[str, Any],
    content: str,
    path_kinds: dict[str, str],
    constant_sources: set[str],
    content_hash_to_turns: dict[str, list[int]],
    turn: int,
) -> str | None:
    """Решаем, по какому правилу свернуть это role=tool сообщение."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("tool") != "read_file":
        # Intra-history работает для любых идентичных наблюдений, не только read.
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        turns = content_hash_to_turns.get(digest, [])
        if len(turns) > 1 and turn != turns[-1]:
            return "intra_history"
        return None
    # read_file: свернём, если путь уже представлен постоянным фрагментом.
    for path, _kind in path_kinds.items():
        if path in constant_sources:
            return "path_match"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    turns = content_hash_to_turns.get(digest, [])
    if len(turns) > 1 and turn != turns[-1]:
        return "intra_history"
    return None


def _dedup_summary(
    message: dict[str, Any],
    content: str,
    rule: str,
    read_path: str | None = None,
) -> str:
    """Компактная сводка свёрнутого наблюдения в том же ok/tool/summary формате.

    Для read_file строим дайджест-указатель (путь + диапазон строк), чтобы
    модель сохраняла знание о прочитанном, а не теряла его в обрезке.
    """

    tool = ""
    is_read_file = False
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            tool = str(payload.get("tool") or "")
            is_read_file = tool == "read_file"
    except json.JSONDecodeError:
        pass
    if is_read_file:
        return digest_read_file(content, read_path)
    ok_value = True
    summary = ""
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            ok_value = bool(payload.get("ok", True))
            summary = str(payload.get("summary") or "")
    except json.JSONDecodeError:
        pass
    compact = {"ok": ok_value, "tool": tool, "summary": summary}
    compact["_context_deduped"] = True
    compact["dedup_rule"] = rule
    if rule == "path_match":
        compact["note"] = "content already in context as a constant fragment"
    elif rule == "intra_history":
        compact["note"] = "duplicate of a later observation"
    return json.dumps(compact, ensure_ascii=False)
