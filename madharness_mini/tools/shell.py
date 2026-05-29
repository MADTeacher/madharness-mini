"""Инструмент запуска разрешённой shell-команды в workspace."""

import shlex
import subprocess
from typing import Any

from ..utils import clipped, fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec


def run_shell(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Выполняем одну команду в workspace или подкаталоге после проверки Policy."""

    command = args["command"]
    allowed, reason = ctx.policy.shell_allowed(command)
    if not allowed:
        return fail("run_shell", reason, command=command)
    cwd_raw = args.get("cwd", ".")
    cwd, err = ctx.policy.safe_path(cwd_raw)
    if err or not cwd:
        return fail("run_shell", err or f"invalid cwd: {cwd_raw}", command=command)
    if not cwd.is_dir():
        return fail("run_shell", f"cwd is not a directory: {cwd_raw}", command=command)
    if ctx.trace and ctx.skill_runtime:
        event = ctx.skill_runtime.resource_event(cwd)
        if event:
            ctx.trace.write("skill_resource_used", tool="run_shell", **event)
    proc = subprocess.run(
        shlex.split(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=60,
    )
    try:
        cwd_display = str(cwd.relative_to(ctx.cfg.root))
    except ValueError:
        cwd_display = str(cwd)
    return ok(
        "run_shell",
        f"exit code {proc.returncode}",
        command=command,
        cwd=cwd_display or ".",
        returncode=proc.returncode,
        stdout=clipped(proc.stdout),
        stderr=clipped(proc.stderr),
    )


RUN_SHELL_DESCRIPTION = """Run one allowed command in the workspace.

Use this for tests, builds, safe repository inspection, and documented skill
scripts. The command runs from the workspace root by default, or from a
workspace-relative cwd such as a skill root when cwd is provided, and times out
after 60 seconds. It must be a single command: shell control operators such as
|, >, <, &&, ||, and ; are denied, and risky commands such as sudo, curl, wget,
ssh, scp, chmod 777, mkfs, dd, and rm -rf are blocked by policy.
Do not use run_shell to edit files; use apply_patch for precise edits and write_file
only for deliberate full rewrites.
"""

RUN_SHELL_SPEC = ToolSpec(
    "run_shell",
    RUN_SHELL_DESCRIPTION,
    obj(
        {
            "command": strp(
                req=True,
                desc="Single safe command with arguments, run from the workspace root; no shell control operators and no file-editing scripts.",
            ),
            "cwd": strp(
                ".",
                "Workspace-relative directory to run from; use a skill root only for documented bundled scripts.",
            ),
        },
        ["command"],
    ),
    run_shell,
)
