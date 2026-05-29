"""Публичные режимы ask (один ответ) и run (цикл с инструментами)."""

from __future__ import annotations

import time
from typing import Any

from .config import Config
from .context import ContextProvider
from .context.bootstrap import base_context
from .model import ModelClient
from .model_loop import (
    apply_hidden_observation_effects,
    call_model_with_rate_limit_retry,
    run_model_loop,
    safe_context_report,
)
from .mcp import McpToolProvider
from .skills import (
    SkillCatalogProvider,
    SkillRuntime,
    SkillToolProvider,
    discover_skills,
    find_explicit_skill_selection,
)
from .subagents import OrchestratorToolProvider, discover_subagents
from .subagents.orchestration import (
    parent_allowed_tools,
    required_orchestration_fragment,
    resolve_orchestration_mode,
)
from .subagents.runner import run_subagent
from .tools import ToolRegistry
from .trace import Trace

# Старые тесты и внешние harness-патчи могут подменять `loop.time.sleep`;
# модуль `time` общий для Python-процесса, поэтому retry в `model_loop` это увидит.
_SLEEP_PATCH_COMPAT = time


def ask(task: str, cfg: Config) -> tuple[str, Any]:
    """Один запрос к модели без инструментов; пишем трассу в JSONL.

    Возвращаем текст ответа и путь к файлу трассы.
    """

    trace = Trace(cfg, "ask")
    context = base_context(cfg, task)
    try:
        messages = context.messages()
    except RuntimeError as exc:
        trace.write(
            "context_error",
            error=str(exc),
            context_report=safe_context_report(context),
        )
        trace.write("session_end", result=f"error: {exc}")
        raise
    trace.write("model_call_started", tools_count=0, context_report=context.report())
    try:
        raw = call_model_with_rate_limit_retry(ModelClient(cfg), trace, messages)
    except RuntimeError as exc:
        trace.write("model_error", error=str(exc))
        trace.write("session_end", result=f"error: {exc}")
        raise
    trace.write("model_call_finished", raw=raw)
    content = raw["choices"][0]["message"].get("content") or ""
    trace.write("session_end", result=content)
    return content, trace.path


def run_agent(
    task: str,
    cfg: Config,
    *,
    orchestration_mode: str | None = None,
) -> tuple[str, Any]:
    """Агентский цикл до финального текста или исчерпания max_turns.

    Здесь остаётся сборка запуска: discovery расширений, registry инструментов
    и стартовый контекст. Сам turn loop и делегация вынесены в профильные модули.
    """

    trace = Trace(cfg, "run")
    client = ModelClient(cfg)
    skill_index = discover_skills(cfg)
    subagent_index = discover_subagents(cfg)
    orchestration = resolve_orchestration_mode(
        cfg,
        task,
        override=orchestration_mode,
    )
    trace.write(
        "orchestration_mode",
        configured=orchestration["configured"],
        effective=orchestration["effective"],
        source=orchestration["source"],
        requested_by_task=orchestration["requested_by_task"],
        legacy_enabled=bool(cfg.data.get("orchestration_enabled", True)),
    )
    trace.write(
        "skills_discovered",
        count=len(skill_index.skills),
        names=skill_index.names(),
        diagnostics=[
            diagnostic.as_dict(cfg.root) for diagnostic in skill_index.diagnostics
        ],
    )
    trace.write(
        "subagents_discovered",
        count=len(subagent_index.subagents),
        names=subagent_index.names(),
        diagnostics=[diagnostic.as_dict() for diagnostic in subagent_index.diagnostics],
    )
    explicit_skills = find_explicit_skill_selection(task, set(skill_index.skills))
    if explicit_skills.unknown:
        names = ", ".join(explicit_skills.unknown)
        result = f"error: unknown skill: {names}"
        trace.write("session_end", result=result)
        raise RuntimeError(f"unknown skill: {names}")

    skill_runtime = SkillRuntime(cfg, skill_index)
    context_providers: list[ContextProvider] = []
    tool_providers = []
    if explicit_skills.names:
        trace.write("skills_auto_selection_disabled", reason="explicit skill marker")
    else:
        context_providers.append(SkillCatalogProvider(skill_index, cfg.root))
        tool_providers.append(SkillToolProvider(skill_runtime))
    if orchestration["effective"] != "off":
        tool_providers.append(
            OrchestratorToolProvider(
                subagent_index,
                lambda _tool_ctx, subagent, args: run_subagent(
                    cfg,
                    client,
                    trace,
                    subagent,
                    args,
                ),
            )
        )
    tool_providers.append(McpToolProvider())

    tools_registry = ToolRegistry(
        cfg,
        providers=tool_providers,
        trace=trace,
        skill_runtime=skill_runtime,
        allowed_tools=parent_allowed_tools(orchestration["effective"]),
    )
    try:
        context = base_context(cfg, task, providers=context_providers)
        if orchestration["effective"] == "required":
            context.add_fragment(required_orchestration_fragment())
        for name in explicit_skills.names:
            obs = skill_runtime.activate(name, "explicit")
            if not obs.get("ok"):
                result = f"error: {obs.get('summary')}"
                trace.write("session_end", result=result)
                raise RuntimeError(str(obs.get("summary")))
            apply_hidden_observation_effects(context, trace, obs)

        loop_result = run_model_loop(
            client,
            trace,
            context,
            tools_registry,
            int(cfg.data["max_turns"]),
        )
        return str(loop_result["result"]), trace.path
    finally:
        tools_registry.close()
