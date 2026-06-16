"""Маскировка секретов и приватных значений в отчетах тепловой карты."""

from __future__ import annotations

import re

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|ghp|gho|xox[baprs])-?[A-Za-z0-9_=-]{12,}\b"),
]


def redact_text(text: str, *, limit: int = 300) -> tuple[str, bool]:
    """Маскируем секреты и возвращаем короткий excerpt."""

    redacted = text
    found = False
    for pattern in SECRET_PATTERNS:
        redacted, count = pattern.subn(_replacement, redacted)
        found = found or bool(count)
    if len(redacted) > limit:
        redacted = redacted[:limit] + f"\n...[clipped {len(redacted) - limit} chars]"
    return redacted, found


def contains_secret(text: str) -> bool:
    """Проверяем, похож ли текст на секрет."""

    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _replacement(match: re.Match[str]) -> str:
    """Сохраняем имя поля, но скрываем значение."""

    if match.lastindex:
        return f"{match.group(1)}<redacted>"
    return "<redacted-secret>"
