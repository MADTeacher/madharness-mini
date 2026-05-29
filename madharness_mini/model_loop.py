"""Общий цикл model/tool вызовов для parent и субагентов."""

from __future__ import annotations

import time
from typing import Any

from .context import ContextManager
from .model import ModelClient, ModelRateLimitError
from .tools import ToolRegistry
from .trace import Trace
from .utils import fail, parse_tool_args

# При 429 ждём Retry-After, но не дольше этой границы (секунды).
RATE_LIMIT_RETRY_MAX_SECONDS = 60


def call_model_with_rate_limit_retry(
    client: ModelClient,
    trace: Trace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **trace_data: Any,
) -> dict[str, Any]:
    """Зовём модель; при коротком 429 один раз ждём и повторяем запрос.

    Длинный Retry-After пробрасываем наверх — пользователь увидит ошибку в CLI.
    """

    try:
        return client.chat(messages, tools)
    except ModelRateLimitError as exc:
        wait_seconds = exc.retry_after_seconds
        if wait_seconds is not None and 0 < wait_seconds <= RATE_LIMIT_RETRY_MAX_SECONDS:
            trace.write(
                "model_rate_limit_retry",
                **trace_data,
                status=exc.status,
                retry_after=exc.retry_after,
                retry_after_seconds=wait_seconds,
            )
            time.sleep(wait_seconds)
            return client.chat(messages, tools)
        raise


def run_model_loop(
    client: ModelClient,
    trace: Trace,
    context: ContextManager,
    tools_registry: ToolRegistry,
    max_turns: int,
    *,
    stop_on_user_input: bool = False,
) -> dict[str, Any]:
    """Ведём модель через ходы assistant/tool до финального результата."""

    for turn in range(max_turns):
        tool_schemas = tools_registry.schemas()
        try:
            messages = context.messages(tool_schemas)
        except RuntimeError as exc:
            trace.write(
                "context_error",
                turn=turn,
                error=str(exc),
                context_report=safe_context_report(context),
            )
            trace.write("session_end", result=f"error: {exc}")
            raise
        trace.write(
            "model_call_started",
            turn=turn,
            tools_count=len(tool_schemas),
            context_report=context.report(),
        )
        try:
            raw = call_model_with_rate_limit_retry(
                client, trace, messages, tool_schemas, turn=turn
            )
        except RuntimeError as exc:
            trace.write("model_error", turn=turn, error=str(exc))
            trace.write("session_end", result=f"error: {exc}")
            raise
        message = raw["choices"][0]["message"]
        trace.write("model_call_finished", turn=turn, message=message)
        context.record_assistant(message)
        calls = message.get("tool_calls") or []
        if not calls:
            result = message.get("content") or ""
            trace.write("session_end", result=result)
            return {"status": "done", "result": result, "turns": turn + 1}
        for call in calls:
            try:
                name, args = parse_tool_args(call)
                obs = tools_registry.call(name, args)
            except Exception as exc:
                name, args = "tool_call", {}
                obs = fail(name, f"invalid tool call: {exc}")
            followup_messages = obs.pop("_followup_messages", [])
            subagent_stop = obs.pop("_subagent_stop", "")
            apply_hidden_observation_effects(context, trace, obs)
            trace.write("tool_observation", tool=name, args=args, observation=obs)
            if stop_on_user_input and subagent_stop == "needs_user_input":
                trace.write(
                    "session_end",
                    result=f"needs_user_input: {obs.get('question', '')}",
                )
                return {
                    "status": "needs_user_input",
                    "result": obs.get("question", ""),
                    "observation": obs,
                    "turns": turn + 1,
                }
            context.record_tool_result(call, obs, followup_messages)
            if is_parent_user_input_request(obs):
                result = render_user_input_request(obs)
                trace.write(
                    "user_input_requested",
                    subagent=obs.get("subagent", ""),
                    question=obs.get("question", ""),
                    options=obs.get("options", []),
                    reason=obs.get("reason", ""),
                    subagent_trace_id=obs.get("subagent_trace_id", ""),
                    subagent_trace_path=obs.get("subagent_trace_path", ""),
                )
                trace.write("session_end", result=result)
                return {
                    "status": "needs_user_input",
                    "result": result,
                    "observation": obs,
                    "turns": turn + 1,
                }
    result = "Agent stopped: max_turns exceeded."
    trace.write("session_end", result=result)
    return {"status": "max_turns", "result": result, "turns": max_turns}


def apply_hidden_observation_effects(
    context: ContextManager,
    trace: Trace,
    observation: dict[str, Any],
) -> None:
    """Применяем служебные эффекты tool observation, не отправляя их модели."""

    fragments = observation.pop("_context_fragments", [])
    for fragment in fragments:
        context.add_fragment(fragment)
    skill_event = observation.pop("_skill_event", None)
    if skill_event:
        trace.write("skill_activated", **skill_event)


def is_parent_user_input_request(observation: dict[str, Any]) -> bool:
    """Понимаем, что делегация просит остановить `run` и спросить пользователя."""

    return (
        observation.get("tool") == "delegate_task"
        and observation.get("status") == "needs_user_input"
        and bool(str(observation.get("question") or "").strip())
    )


def render_user_input_request(observation: dict[str, Any]) -> str:
    """Печатаем вопрос субагента напрямую пользователю без ещё одного model call."""

    subagent = str(observation.get("subagent") or "subagent").strip()
    question = str(observation.get("question") or "").strip()
    reason = str(observation.get("reason") or "").strip()
    options = observation.get("options") or []
    lines = [f"{subagent} просит уточнение:", "", question]
    cleaned_options = [str(item).strip() for item in options if str(item).strip()]
    if cleaned_options:
        lines.extend(["", "Варианты:"])
        lines.extend(f"{index}. {item}" for index, item in enumerate(cleaned_options, 1))
    if reason:
        lines.extend(["", f"Причина: {reason}"])
    lines.extend(
        [
            "",
            "Ответьте на вопрос и повторите команду `run` с выбранным решением в задаче.",
        ]
    )
    return "\n".join(lines)


def safe_context_report(context: ContextManager) -> dict[str, Any]:
    """Пишем context error в trace, даже если повторная сборка отчёта тоже падает."""

    try:
        return context.report()
    except Exception as exc:
        return {"error": str(exc)}
