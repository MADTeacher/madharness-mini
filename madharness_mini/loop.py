from __future__ import annotations

import json
from typing import Any

from .config import Config
from .instructions import load_agents_md, load_prompt
from .model import ModelClient
from .tools import ToolRegistry
from .trace import Trace
from .utils import fail, parse_tool_args


def base_messages(cfg: Config, task: str) -> list[dict[str, Any]]:
    instructions = load_agents_md(cfg.root, cfg.cwd)
    system = load_prompt("system")
    if instructions:
        system = f"{system}\n\n{instructions}"
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


def ask(task: str, cfg: Config) -> tuple[str, Any]:
    trace = Trace(cfg, "ask")
    messages = base_messages(cfg, task)
    trace.write("model_call_started", tools_count=0)
    try:
        raw = ModelClient(cfg).chat(messages)
    except RuntimeError as exc:
        trace.write("model_error", error=str(exc))
        trace.write("session_end", result=f"error: {exc}")
        raise
    trace.write("model_call_finished", raw=raw)
    content = raw["choices"][0]["message"].get("content") or ""
    trace.write("session_end", result=content)
    return content, trace.path


def run_agent(task: str, cfg: Config) -> tuple[str, Any]:
    trace = Trace(cfg, "run")
    client = ModelClient(cfg)
    registry = ToolRegistry(cfg)
    messages = base_messages(cfg, task)
    for turn in range(int(cfg.data["max_turns"])):
        trace.write("model_call_started", turn=turn, tools_count=len(registry.tools))
        try:
            raw = client.chat(messages, registry.schemas())
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
