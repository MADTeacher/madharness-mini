"""Общий цикл model/tool вызовов для режима run."""

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
            trace.write("tool_observation", tool=name, args=args, observation=obs)
            context.record_tool_result(call, obs, followup_messages)
    result = "Agent stopped: max_turns exceeded."
    trace.write("session_end", result=result)
    return {"status": "max_turns", "result": result, "turns": max_turns}


def safe_context_report(context: ContextManager) -> dict[str, Any]:
    """Пишем context error в trace, даже если повторная сборка отчёта тоже падает."""

    try:
        return context.report()
    except Exception as exc:
        return {"error": str(exc)}
