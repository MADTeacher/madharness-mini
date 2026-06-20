"""Реализация суммаризатора рассуждений на базе ModelClient.

Слой контекста (`ContextManager`) умеет звать внешний `ReasoningSummarizer`,
когда история превышает токеновый порог, но сам модель не дёргает: это нарушило
бы его изоляцию от Config/ModelClient. Конкретная реализация живёт здесь, в
верхнем слое harness, и внедряется в контекст через bootstrap (инверсия
зависимостей). Один вызов модели сворачивает старые ходы в компактную сводку и
пишет событие trace, чтобы модельный вызов учитывался тепловой картой.
"""

from __future__ import annotations

from typing import Any

from .context.budget import clip_text
from .context.history import HistoryEntry
from .model import ModelClient
from .trace import Trace

# Каждый ход в запрос суммаризатора отдаём усечённым: полная история уже не
# помещается в окно, а для фактической сводки достаточно начала каждого сообщения.
_MESSAGE_CONTENT_LIMIT = 1000

# Фиксированный системный промпт: без вложенных инструкций модели, чтобы свёртка
# оставалась предсказуемой и не подхватывала указания из самой истории.
_SYSTEM_PROMPT = (
    "Сожми историю работы агента в компактную фактическую сводку. Сохрани пути "
    "файлов, принятые решения, открытые проблемы и следующий шаг. Не выдумывай "
    "факты."
)


class ModelReasoningSummarizer:
    """Сворачивает старые ходы истории в текст одним вызовом ModelClient.

    Реализует протокол `ReasoningSummarizer`: не мутирует entries, возвращает
    обновлённую накопительную сводку, а при пустом ответе или любой ошибке —
    пустую строку (детерминированный fallback «не сворачивать»). Перед возвратом
    всегда пишет событие trace `context_summary`.
    """

    def __init__(self, client: ModelClient, trace: Trace):
        self.client = client
        self.trace = trace

    def summarize(self, entries: list[HistoryEntry], previous_summary: str) -> str:
        """Возвращаем сводку по ходам entries и предыдущей сводке.

        Строим компактные сообщения (роль + усечённый контент), делаем один
        вызов `client.chat` без tools и достаём content ответа. Пустой ответ или
        исключение трактуем как «не сворачивать» и возвращаем "".
        """

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._serialize(entries, previous_summary)},
        ]
        result = ""
        try:
            raw = self.client.chat(messages)
            content = raw["choices"][0]["message"]["content"]
            if isinstance(content, str):
                result = content
        except Exception:
            # Fallback: при любом сбое суммаризатора контекст остаётся прежним.
            result = ""
        self.trace.write("context_summary", chars=len(result))
        return result

    def _serialize(self, entries: list[HistoryEntry], previous_summary: str) -> str:
        """Сериализуем предыдущую сводку и ходы в один текст для запроса модели."""

        lines: list[str] = []
        if previous_summary:
            lines.append("Предыдущая сводка:")
            lines.append(previous_summary)
            lines.append("")
        lines.append("Ходы для свёртки:")
        for entry in entries:
            for message in entry.rendered_messages():
                role = str(message.get("role") or "")
                content = _message_text(message.get("content"))
                lines.append(f"[{role}] {clip_text(content, _MESSAGE_CONTENT_LIMIT)}")
        return "\n".join(lines)


def _message_text(content: Any) -> str:
    """Приводим контент сообщения к строке: списки/None превращаем в текст."""

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    # Мультимодальный контент (например, image_url) сериализуем грубо: для
    # фактической сводки важен сам факт, а не точная структура вложения.
    return str(content)
