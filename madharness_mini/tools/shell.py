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


RUN_SHELL_DESCRIPTION = """Run one allowed command in the workspace.

Use this for tests, builds, and safe repository inspection. The command runs
with cwd set to the workspace root and times out after 60 seconds. It must be a
single command: shell control operators such as |, >, <, &&, ||, and ; are
denied, and risky commands such as sudo, curl, wget, ssh, scp, chmod 777, mkfs,
dd, and rm -rf are blocked by policy. Do not use run_shell to edit files; use
apply_patch for precise edits and write_file only for deliberate full rewrites.
"""

RUN_SHELL_SPEC = ToolSpec(
    "run_shell",
    RUN_SHELL_DESCRIPTION,
    obj(
        {
            "command": strp(
                req=True,
                desc="Single safe command with arguments, run from the "
                "workspace root; no shell control operators and no "
                "file-editing scripts.",
            )
        },
        ["command"],
    ),
    run_shell,
)
