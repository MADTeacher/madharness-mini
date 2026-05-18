"""Исполнение команд `ask` и `run` через модель."""

from __future__ import annotations

import json
import time
from typing import Any

from .config import Config
from .instructions import load_prompt
from .model import ModelClient, ModelRateLimitError
from .tools import ToolRegistry
from .trace import Trace
from .utils import fail, parse_tool_args

RATE_LIMIT_RETRY_MAX_SECONDS = 60


def base_messages(cfg: Config, task: str) -> list[dict[str, Any]]:
    """Подготовить начальные сообщения модели для задачи пользователя."""

    system = load_prompt("system")
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


def call_model_with_rate_limit_retry(
    client: ModelClient,
    trace: Trace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **trace_data: Any,
) -> dict[str, Any]:
    """Вызвать модель и один раз переждать короткий `Retry-After` при 429."""

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
    """Выполнить одно обращение к модели без инструментов и записать трассу."""

    trace = Trace(cfg, "ask")
    messages = base_messages(cfg, task)
    trace.write("model_call_started", tools_count=0)
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
    """Вести агентский цикл до финального ответа или исчерпания ходов.

    На каждом ходе модель либо возвращает текст, либо запрашивает один или
    несколько `tool_calls`. Локальные результаты инструментов добавляются в
    историю как JSON-наблюдения.
    """

    trace = Trace(cfg, "run")
    client = ModelClient(cfg)
    registry = ToolRegistry(cfg)
    messages = base_messages(cfg, task)
    for turn in range(int(cfg.data["max_turns"])):
        trace.write("model_call_started", turn=turn, tools_count=len(registry.tools))
        try:
            raw = call_model_with_rate_limit_retry(
                client, trace, messages, registry.schemas(), turn=turn
            )
        except RuntimeError as exc:
            trace.write("model_error", turn=turn, error=str(exc))
            trace.write("session_end", result=f"error: {exc}")
            raise
        message = raw["choices"][0]["message"]
        trace.write("model_call_finished", turn=turn, message=message)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            result = message.get("content") or ""
            trace.write("session_end", result=result)
            return result, trace.path
        for call in calls:
            try:
                name, args = parse_tool_args(call)
                obs = registry.call(name, args)
            except Exception as exc:
                name, args = "tool_call", {}
                obs = fail(name, f"invalid tool call: {exc}")
            trace.write("tool_observation", tool=name, args=args, observation=obs)
            content = json.dumps(obs, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "content": content,
                }
            )
    result = "Agent stopped: max_turns exceeded."
    trace.write("session_end", result=result)
    return result, trace.path
