"""Стартовая сборка контекста для режимов ask/run."""

from __future__ import annotations

from collections.abc import Iterable

from ..config import Config
from ..instructions import load_project_instructions, load_prompt
from .fragments import ContextFragment, ContextProvider
from .manager import ContextManager
from .summary import ReasoningSummarizer


def base_context(
    cfg: Config,
    task: str,
    providers: Iterable[ContextProvider] | None = None,
    *,
    max_tokens: int | None = None,
    summarizer: ReasoningSummarizer | None = None,
    summary_trigger_tokens: int | None = None,
) -> ContextManager:
    """Готовим слой контекста для ask/run: system, AGENTS.md и задача.

    Сам ContextManager не читает файлы и не знает про Config. Bootstrap передаёт
    ему уже готовый системный текст, чтобы граница слоя контекста оставалась
    простой. Внешний суммаризатор (DIP) и токеновый порог опциональны: если порог
    не задан явно, берём его из cfg.data (по умолчанию 0 — LLM-свёртка выключена).
    """

    context = ContextManager(
        task,
        max_tokens=(
            max_tokens
            if max_tokens is not None
            else int(cfg.data.get("context_max_tokens", 60000))
        ),
        keep_recent_turns=int(cfg.data.get("context_keep_recent_turns", 3)),
        summarize_after_turns=int(cfg.data.get("context_summarize_after_turns", 3)),
        providers=providers,
        # Fan-out защита: чтение контракта не сворачивается, если за ним последовало
        # достаточно правок зависимых файлов в заданном окне ходов.
        contract_protection_turns=int(
            cfg.data.get("context_contract_protection_turns", 12)
        ),
        contract_protection_writes=int(
            cfg.data.get("context_contract_protection_writes", 3)
        ),
        summarizer=summarizer,
        summary_trigger_tokens=(
            summary_trigger_tokens
            if summary_trigger_tokens is not None
            else int(cfg.data.get("context_summary_trigger_tokens", 0))
        ),
    )
    context.add_fragment(
        ContextFragment(
            id="system",
            source="madharness_mini/prompts/system.md",
            text=load_prompt("system"),
            priority=0,
            placement="system",
            authority_level="system",
            context_layer="normative",
            evictability="never",
            stability="stable",
            applicability="active",
            normative_role="safety",
        )
    )
    project_instructions = load_project_instructions(cfg)
    if project_instructions:
        context.add_fragment(
            ContextFragment(
                id="project-instructions",
                source="AGENTS.md",
                text=f"# Project instructions\n\n{project_instructions}",
                priority=1,
                placement="system",
                authority_level="project",
                context_layer="normative",
                evictability="only_after_validation",
                stability="stable",
                applicability="current_project",
                normative_role="workflow",
            )
        )
    return context
