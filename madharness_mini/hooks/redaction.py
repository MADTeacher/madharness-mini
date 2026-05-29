"""Компактное представление payload для пользовательских hooks и trace."""

from __future__ import annotations

from typing import Any

from ..utils import clipped

SENSITIVE_KEYS = {"apikey", "authorization", "password", "secret", "token"}


def compact_payload(
    value: Any,
    *,
    string_limit: int = 2000,
    list_limit: int = 20,
    depth: int = 5,
) -> Any:
    """Обрезаем большие значения и прячем очевидные секреты перед hook-командой.

    Hooks нужны для политики и аудита, но не должны случайно получать огромный
    prompt, base64 или ключи API. Поэтому передаём структуру, похожую на исходную,
    но с ограниченной глубиной и длиной строк.
    """

    if depth <= 0:
        return "<max depth>"
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_sensitive_key(text_key):
                compact[text_key] = "<redacted>"
            else:
                compact[text_key] = compact_payload(
                    item,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    depth=depth - 1,
                )
        return compact
    if isinstance(value, list):
        items = [
            compact_payload(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth - 1,
            )
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            items.append(f"<clipped {len(value) - list_limit} items>")
        return items
    if isinstance(value, tuple):
        return compact_payload(
            list(value),
            string_limit=string_limit,
            list_limit=list_limit,
            depth=depth,
        )
    if isinstance(value, str):
        return clipped(value, string_limit)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return clipped(str(value), string_limit)


def _is_sensitive_key(key: str) -> bool:
    """Отсекаем поля, которые обычно содержат токены или пароли."""

    lowered = key.lower().replace("-", "_")
    parts = [part for part in lowered.split("_") if part]
    return (
        lowered.endswith("api_key")
        or lowered.endswith("_token")
        or "token" in parts
        or any(part in SENSITIVE_KEYS for part in parts)
    )
