"""Режимы ask (один ответ) и run (цикл с инструментами)."""

import time
from typing import Any

from .config import Config
from .context import ContextFragment, ContextManager
from .instructions import load_project_instructions, load_prompt
from .model import ModelClient, ModelRateLimitError
from .tools import ToolRegistry
from .trace import Trace
from .utils import fail, parse_tool_args

# При 429 ждём Retry-After, но не дольше этой границы (секунды).
RATE_LIMIT_RETRY_MAX_SECONDS = 60


def base_context(cfg: Config, task: str) -> ContextManager:
    """Готовим слой контекста для ask/run: system, AGENTS.md и задача.

    Сам ContextManager не читает файлы и не знает про Config. Loop передаёт ему
    уже готовый системный текст, чтобы граница слоя контекста оставалась простой.
    """

    context = ContextManager(
        task,
        max_tokens=int(cfg.data.get("context_max_tokens", 60000)),
        keep_recent_turns=int(cfg.data.get("context_keep_recent_turns", 3)),
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
            context_report=_safe_context_report(context),
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


def run_agent(task: str, cfg: Config) -> tuple[str, Any]:
    """Агентский цикл до финального текста или исчерпания max_turns.

    На каждом ходе модель либо отвечает текстом, либо шлёт tool_calls.
    Результаты инструментов добавляем в историю как role=tool (JSON).
    """

    trace = Trace(cfg, "run")
    client = ModelClient(cfg)
    tools_registry = ToolRegistry(cfg)
    context = base_context(cfg, task)
    for turn in range(int(cfg.data["max_turns"])):
        tool_schemas = tools_registry.schemas()
        try:
            messages = context.messages(tool_schemas)
        except RuntimeError as exc:
            trace.write(
                "context_error",
                turn=turn,
                error=str(exc),
                context_report=_safe_context_report(context),
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
            return result, trace.path
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
    return result, trace.path


def _safe_context_report(context: ContextManager) -> dict[str, Any]:
    """Пишем context error в trace, даже если повторная сборка отчёта тоже падает."""

    try:
        return context.report()
    except Exception as exc:
        return {"error": str(exc)}
