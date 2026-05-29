"""Стартовая сборка контекста для режимов ask/run."""

from __future__ import annotations

from collections.abc import Iterable

from ..config import Config
from ..instructions import load_project_instructions, load_prompt
from .fragments import ContextFragment, ContextProvider
from .manager import ContextManager


def base_context(
    cfg: Config,
    task: str,
    providers: Iterable[ContextProvider] | None = None,
    *,
    max_tokens: int | None = None,
) -> ContextManager:
    """Готовим слой контекста для ask/run: system, AGENTS.md и задача.

    Сам ContextManager не читает файлы и не знает про Config. Bootstrap передаёт
    ему уже готовый системный текст, чтобы граница слоя контекста оставалась
    простой.
    """

    context = ContextManager(
        task,
        max_tokens=(
            max_tokens
            if max_tokens is not None
            else int(cfg.data.get("context_max_tokens", 60000))
        ),
        keep_recent_turns=int(cfg.data.get("context_keep_recent_turns", 3)),
        providers=providers,
    )
    system = load_prompt("system")
    project_instructions = load_project_instructions(cfg)
    if project_instructions:
        system = f"{system}\n\n# Project instructions\n\n{project_instructions}"
    context.add_fragment(
        ContextFragment(
            id="system",
            source="madharness_mini/prompts/system.md",
            text=system,
            priority=0,
            placement="system",
        )
    )
    return context
