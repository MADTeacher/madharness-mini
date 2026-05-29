"""Tool providers для родительской делегации и вопросов пользователю."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..tools.context import ToolContext
from ..tools.specs import ToolSpec
from ..utils import fail, obj, ok, strp
from .types import Subagent, SubagentIndex

SubagentRunner = Callable[[ToolContext, Subagent, dict[str, Any]], dict[str, Any]]


class OrchestratorToolProvider:
    """Добавляет родительскому агенту инструмент делегирования."""

    def __init__(self, index: SubagentIndex, runner: SubagentRunner):
        self.index = index
        self.runner = runner

    def specs(self, ctx: ToolContext) -> Iterable[ToolSpec]:
        """Возвращаем delegate_task только если в каталоге есть субагенты."""

        if not self.index.subagents:
            return []
        return [
            ToolSpec(
                "delegate_task",
                _delegate_description(self.index),
                obj(
                    {
                        "subagent": {
                            "type": "string",
                            "description": "Built-in role or project-local subagent name.",
                            "enum": self.index.names(),
                        },
                        "task": strp(
                            req=True,
                            desc="Small self-contained task for the subagent.",
                        ),
                        "context": strp(
                            "",
                            "Short parent context, constraints, or decisions the subagent should know.",
                        ),
                        "profile": {
                            "type": "string",
                            "description": "Optional requested profile. read-only can downgrade writable subagents; writable cannot upgrade read-only subagents.",
                            "enum": ["read-only", "writable"],
                        },
                    },
                    ["subagent", "task"],
                ),
                self.delegate,
            )
        ]

    def delegate(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        """Находим субагента и передаём запуск callback из loop.py."""

        name = str(args.get("subagent") or "").strip()
        subagent = self.index.subagents.get(name)
        if not subagent:
            return fail("delegate_task", f"unknown subagent: {name}")
        return self.runner(ctx, subagent, args)


class AskUserToolProvider:
    """Добавляет субагенту tool для контролируемого вопроса пользователю."""

    def specs(self, ctx: ToolContext) -> Iterable[ToolSpec]:
        """Возвращаем ask_user; allow-list registry сам решит, видит ли его роль."""

        return [
            ToolSpec(
                "ask_user",
                "Ask the parent agent to stop this delegation and ask the user a concise question.",
                obj(
                    {
                        "question": strp(
                            req=True,
                            desc="Short question the parent should ask the user.",
                        ),
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                            "description": "Optional 2-3 short options when choices are clear.",
                        },
                        "reason": strp(
                            "",
                            "Why the answer is needed before continuing.",
                        ),
                    },
                    ["question"],
                ),
                ask_user,
            )
        ]


def ask_user(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Останавливаем дочерний loop и возвращаем вопрос родителю."""

    options = args.get("options") or []
    if not isinstance(options, list):
        options = []
    cleaned_options = [str(item).strip() for item in options if str(item).strip()]
    obs = ok(
        "ask_user",
        "user input requested",
        status="needs_user_input",
        question=str(args.get("question") or "").strip(),
        options=cleaned_options[:3],
        reason=str(args.get("reason") or "").strip(),
    )
    obs["_subagent_stop"] = "needs_user_input"
    return obs


def effective_tools(subagent: Subagent, requested_profile: str = "") -> tuple[str, ...]:
    """Считаем итоговый allow-list tools с учётом безопасного downgrade."""

    if requested_profile and requested_profile not in {"read-only", "writable"}:
        raise RuntimeError(f"invalid requested profile: {requested_profile}")
    if requested_profile == "writable" and subagent.profile != "writable":
        raise RuntimeError(f"subagent is not writable: {subagent.name}")
    tools = list(subagent.tools)
    if requested_profile == "read-only":
        denied = {"apply_patch", "write_file", "run_shell"}
        tools = [name for name in tools if name not in denied]
    return tuple(tools)


def _delegate_description(index: SubagentIndex) -> str:
    """Готовим короткое описание delegate_task со списком доступных исполнителей."""

    lines = [
        "Delegate a small task to a built-in or project-local subagent.",
        "The subagent runs in its own context with its own local trace.",
        "Available subagents:",
    ]
    for name in index.names():
        item = index.subagents[name]
        lines.append(f"- {item.name} ({item.profile}): {item.description}")
    return "\n".join(lines)
