"""Запуск делегированного agent loop для markdown-субагентов."""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..context import ContextFragment
from ..context.bootstrap import base_context
from ..hooks import HookManager
from ..model import ModelClient
from ..model_loop import emit_session_error, run_model_loop
from ..tools import ToolRegistry
from ..trace import Trace
from ..utils import fail, ok
from .runtime import summarize_subagent_trace, trace_path_for_observation
from .tools import AskUserToolProvider, effective_tools
from .types import Subagent


def run_subagent(
    cfg: Config,
    client: ModelClient,
    parent_trace: Trace,
    subagent: Subagent,
    args: dict[str, Any],
    hooks: HookManager | None = None,
) -> dict[str, Any]:
    """Запускаем делегированный agent loop с локальным trace и своим allow-list."""

    task = str(args.get("task") or "").strip()
    if not task:
        return fail("delegate_task", "empty subagent task", subagent=subagent.name)
    requested_profile = str(args.get("profile") or "").strip()
    try:
        allowed_tools = effective_tools(subagent, requested_profile)
    except RuntimeError as exc:
        return fail("delegate_task", str(exc), subagent=subagent.name)
    sub_trace = parent_trace.child("subagent", f"subagent-{subagent.name}")
    sub_hooks = hooks.with_trace(sub_trace) if hooks else None
    if sub_hooks:
        sub_hooks.emit(
            "session_start",
            kind="subagent",
            data={
                "subagent": subagent.name,
                "task_preview": task[:1000],
                "parent_trace_id": parent_trace.id,
            },
        )
    trace_path = trace_path_for_observation(sub_trace.path, cfg.cwd)
    parent_trace.write(
        "subagent_started",
        name=subagent.name,
        profile=requested_profile or subagent.profile,
        trace_id=sub_trace.id,
        trace_path=trace_path,
    )
    try:
        loop_result = _run_subagent_loop(
            cfg,
            client,
            sub_trace,
            subagent,
            task,
            str(args.get("context") or "").strip(),
            allowed_tools,
            sub_hooks,
        )
    except RuntimeError as exc:
        parent_trace.write(
            "subagent_failed",
            name=subagent.name,
            trace_id=sub_trace.id,
            trace_path=trace_path,
            error=str(exc),
        )
        if sub_hooks:
            emit_session_error(sub_hooks, "subagent", exc)
        return fail(
            "delegate_task",
            f"subagent failed: {exc}",
            subagent=subagent.name,
            subagent_trace_id=sub_trace.id,
            subagent_trace_path=trace_path,
        )
    summary = summarize_subagent_trace(sub_trace.path)
    parent_trace.write(
        "subagent_finished",
        name=subagent.name,
        status=loop_result["status"],
        trace_id=sub_trace.id,
        trace_path=trace_path,
        turns=loop_result["turns"],
        trace_summary=summary["summary"],
    )
    base = {
        "subagent": subagent.name,
        "status": loop_result["status"],
        "turns": loop_result["turns"],
        "subagent_trace_id": sub_trace.id,
        "subagent_trace_path": trace_path,
        "trace_summary": summary["summary"],
        "changed_files": summary["changed_files"],
    }
    if loop_result["status"] == "done":
        return ok(
            "delegate_task",
            f"{subagent.name} finished in {loop_result['turns']} turns",
            answer=loop_result["result"],
            **base,
        )
    if loop_result["status"] == "needs_user_input":
        observation = loop_result.get("observation") or {}
        return ok(
            "delegate_task",
            f"{subagent.name} needs user input",
            question=observation.get("question", ""),
            options=observation.get("options", []),
            reason=observation.get("reason", ""),
            **base,
        )
    return fail(
        "delegate_task",
        str(loop_result["result"]),
        **base,
    )


def _run_subagent_loop(
    cfg: Config,
    client: ModelClient,
    trace: Trace,
    subagent: Subagent,
    task: str,
    parent_context: str,
    allowed_tools: tuple[str, ...],
    hooks: HookManager | None = None,
) -> dict[str, Any]:
    """Готовим контекст и tools одного субагента, затем запускаем общий loop."""

    context_max_tokens = subagent.context_max_tokens or int(
        cfg.data.get("subagent_context_max_tokens")
        or cfg.data.get("context_max_tokens", 60000)
    )
    delegated_task = task
    if parent_context:
        delegated_task = f"{task}\n\n# Parent context\n\n{parent_context}"
    context = base_context(cfg, delegated_task, max_tokens=context_max_tokens)
    context.add_fragment(
        ContextFragment(
            id=f"subagent:{subagent.name}",
            source=subagent.location,
            text=_render_subagent_prompt(subagent, allowed_tools),
            priority=5,
            placement="system",
        )
    )
    registry = ToolRegistry(
        cfg,
        providers=[AskUserToolProvider()],
        trace=trace,
        allowed_tools=allowed_tools,
        writable_suffixes=_subagent_writable_suffixes(subagent),
        write_scope_description=_subagent_write_scope_description(subagent),
    )
    try:
        max_turns = subagent.max_turns or int(cfg.data.get("subagent_max_turns", 10))
        return run_model_loop(
            client,
            trace,
            context,
            registry,
            max_turns,
            stop_on_user_input=True,
            hooks=hooks,
            kind="subagent",
        )
    finally:
        registry.close()


def _render_subagent_prompt(subagent: Subagent, allowed_tools: tuple[str, ...]) -> str:
    """Добавляем к prompt субагента его видимую конфигурацию."""

    tools = ", ".join(allowed_tools) if allowed_tools else "none"
    return (
        f"# Subagent: {subagent.name}\n\n"
        f"description: {subagent.description}\n"
        f"profile: {subagent.profile}\n"
        f"tools: {tools}\n"
        f"source: {subagent.location}\n\n"
        f"{subagent.prompt}"
    )


def _subagent_writable_suffixes(subagent: Subagent) -> tuple[str, ...] | None:
    """Ограничиваем planner markdown-файлами с планом, не трогая остальные роли."""

    if subagent.name == "planner":
        return (".md",)
    return None


def _subagent_write_scope_description(subagent: Subagent) -> str:
    """Даём инструментам понятное сообщение об ограничении записи."""

    if subagent.name == "planner":
        return "planner may write only Markdown plan files (.md)"
    return ""
