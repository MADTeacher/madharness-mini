"""Публичные режимы ask (один ответ) и run (цикл с инструментами)."""

from __future__ import annotations

import time
from typing import Any

from .config import Config
from .context.bootstrap import base_context
from .model import ModelClient
from .model_loop import (
    call_model_with_rate_limit_retry,
    run_model_loop,
    safe_context_report,
)
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


def run_agent(task: str, cfg: Config) -> tuple[str, Any]:
    """Агентский цикл до финального текста или исчерпания max_turns."""

    trace = Trace(cfg, "run")
    client = ModelClient(cfg)
    tools_registry = ToolRegistry(cfg)
    context = base_context(cfg, task)
    loop_result = run_model_loop(
        client,
        trace,
        context,
        tools_registry,
        int(cfg.data["max_turns"]),
    )
    return str(loop_result["result"]), trace.path
