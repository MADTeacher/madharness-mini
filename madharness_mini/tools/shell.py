"""Инструмент запуска разрешённой shell-команды в workspace."""

import shlex
import subprocess
from typing import Any

from ..utils import clipped, fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec


def run_shell(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Выполняем одну команду в cfg.root после проверки Policy."""

    command = args["command"]
    allowed, reason = ctx.policy.shell_allowed(command)
    if not allowed:
        return fail("run_shell", reason, command=command)
    proc = subprocess.run(
        shlex.split(command),
        cwd=ctx.cfg.root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return ok(
        "run_shell",
        f"exit code {proc.returncode}",
        command=command,
        returncode=proc.returncode,
        stdout=clipped(proc.stdout),
        stderr=clipped(proc.stderr),
    )


RUN_SHELL_SPEC = ToolSpec(
    "run_shell",
    "Run a safe command in the workspace.",
    obj({"command": strp(req=True)}, ["command"]),
    run_shell,
)
