"""Режимы оркестрации parent-агента и делегированных субагентов."""

from __future__ import annotations

from typing import Any

from ..config import Config, ORCHESTRATION_MODE_VALUES
from ..context import ContextFragment

# Слова, которыми пользователь обычно явно просит включить слой делегации.
ORCHESTRATION_REQUEST_MARKERS = (
    "delegate_task",
    "orchestrate",
    "orchestration",
    "subagent",
    "subagents",
    "используй субагент",
    "использовать субагент",
    "субагент",
    "субагенты",
    "оркестр",
    "делегируй",
    "делегировать",
)

# В строгом режиме parent остаётся координатором: читает контекст и делегирует.
ORCHESTRATION_REQUIRED_PARENT_TOOLS = (
    "list_files",
    "read_file",
    "search_code",
    "delegate_task",
)


def resolve_orchestration_mode(
    cfg: Config,
    task: str,
    *,
    override: str | None = None,
) -> dict[str, Any]:
    """Сводим config/env/CLI к фактическому режиму оркестрации.

    Старое поле `orchestration_enabled=false` сохраняет смысл выключателя,
    если новый mode оставлен в `auto`. CLI override считается явным выбором
    и поэтому не блокируется старым выключателем.
    """

    source = "config"
    configured = (cfg.data.get("orchestration_mode") or "auto").strip().lower()
    if override:
        configured = override.strip().lower()
        source = "cli"
    if configured not in ORCHESTRATION_MODE_VALUES:
        allowed = ", ".join(sorted(ORCHESTRATION_MODE_VALUES))
        raise RuntimeError(
            f"invalid orchestration mode: {configured}; allowed: {allowed}"
        )
    if (
        not override
        and not cfg.data.get("orchestration_enabled", True)
        and configured == "auto"
    ):
        configured = "off"
        source = "legacy orchestration_enabled"
    requested_by_task = task_requests_orchestration(task)
    effective = configured
    if configured == "requested" and not requested_by_task:
        effective = "off"
    return {
        "configured": configured,
        "effective": effective,
        "source": source,
        "requested_by_task": requested_by_task,
    }


def task_requests_orchestration(task: str) -> bool:
    """Грубый, но понятный триггер для режима `requested`."""

    lowered = task.casefold()
    return any(marker in lowered for marker in ORCHESTRATION_REQUEST_MARKERS)


def parent_allowed_tools(mode: str) -> tuple[str, ...] | None:
    """В обычных режимах parent свободен, а `required` делает его координатором."""

    if mode == "required":
        return ORCHESTRATION_REQUIRED_PARENT_TOOLS
    return None


def required_orchestration_fragment() -> ContextFragment:
    """Добавляем parent-агенту явное правило строгой оркестрации."""

    return ContextFragment(
        id="orchestration:required",
        source="runtime",
        text=(
            "Оркестрация обязательна для этого запуска. "
            "Ведущий агент координирует работу и делегирует исследование, "
            "планирование, реализацию, проверку или уточнение через "
            "`delegate_task`. Не выполняй файловые правки или shell-проверки "
            "самостоятельно; поручай их подходящим writable/read-only "
            "субагентам."
        ),
        priority=4,
        placement="system",
    )
