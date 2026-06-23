"""Инструмент запуска разрешённой shell-команды в workspace."""

import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from ..utils import MAX_OUTPUT, clipped, clip_tail, fail, obj, ok, strp
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
    proc = _run_command(ctx, command, cwd)
    try:
        cwd_display = str(cwd.relative_to(ctx.cfg.root))
    except ValueError:
        cwd_display = str(cwd)
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    full_output_enabled = bool(ctx.cfg.data.get("context_shell_full_output", True))
    if full_output_enabled:
        stdout_view, stdout_truncated, stdout_full_path = _clip_with_full_output(
            ctx, stdout
        )
        stderr_view, stderr_truncated, stderr_full_path = _clip_with_full_output(
            ctx, stderr
        )
    else:
        # Прежнее поведение: голова вывода через clipped, полный вывод теряется.
        stdout_view, stdout_truncated, stdout_full_path = clipped(stdout), False, None
        stderr_view, stderr_truncated, stderr_full_path = clipped(stderr), False, None
    observation = ok(
        "run_shell",
        f"exit code {proc.returncode}",
        command=command,
        cwd=cwd_display or ".",
        returncode=proc.returncode,
        stdout=stdout_view,
        stderr=stderr_view,
    )
    if stdout_truncated:
        observation["stdout_truncated"] = True
        observation["stdout_original_chars"] = len(stdout)
        if stdout_full_path is not None:
            observation["stdout_full_path"] = stdout_full_path
    if stderr_truncated:
        observation["stderr_truncated"] = True
        observation["stderr_original_chars"] = len(stderr)
        if stderr_full_path is not None:
            observation["stderr_full_path"] = stderr_full_path
    return observation


def _clip_with_full_output(
    ctx: ToolContext, text: str
) -> tuple[str, bool, str | None]:
    """Хвост вывода + сохранение полного текста во временный файл при обрезке.

    Возвращает (текст для модели, признак обрезки, путь к файлу или None). Если
    обрезки нет — путь не нужен, текст возвращается как есть. Полный вывод пишем
    в state_dir/tool_outputs/, который гарантирует ensure_dirs. Модель может
    перечитать полный файл через read_file, если ей нужен контекст от начала.
    """

    view, truncated = clip_tail(text, MAX_OUTPUT)
    if not truncated:
        return view, False, None
    outputs_dir = ctx.cfg.state_dir / "tool_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    file_path = outputs_dir / f"shell-{uuid.uuid4().hex[:8]}.log"
    try:
        file_path.write_text(text, encoding="utf-8")
    except OSError:
        # Не удалось сохранить полный вывод — отдаём хвост без пути. Модель видит
        # признак обрезки, но перечитать уже не сможет. Лучше, чем ронять команду.
        return view, True, None
    return view, True, str(file_path)


def _run_command(
    ctx: ToolContext, command: str, cwd: Path
) -> subprocess.CompletedProcess:
    """Запускаем команду одним shot-ом: через /bin/sh -c или через execve.

    Режим выбираем по allow_shell_interpreter. Policy уже гарантирует, что в
    режиме с интерпретатором в команде нет разрушительных команд, backticks,
    brace expansion и редиректов в файлы — здесь только механика запуска.
    В старом режиме (без интерпретатора) операторы запрещены в policy, поэтому
    shlex.split безопасен: режет на литеральные аргументы.
    """

    use_shell = bool(ctx.cfg.data.get("allow_shell_interpreter", True))
    if use_shell:
        return subprocess.run(
            command,
            shell=True,
            executable="/bin/sh",
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=60,
        )
    return subprocess.run(
        shlex.split(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=60,
    )


RUN_SHELL_DESCRIPTION = """Run one allowed command in the workspace.

Use this for tests, builds, safe repository inspection, and documented skill
scripts. The command runs from the workspace root by default, or from a
workspace-relative cwd such as a skill root when cwd is provided, and times out
after 60 seconds.

By default the command runs through /bin/sh -c, so shell control operators and
output control work: you may chain commands with && and ;, pipe with |, and
discard output with 2>&1 and >/dev/null (for example `node --version &&
npm --version`, `npx vitest run 2>&1`). The setting allow_shell_interpreter
controls this: when it is false, the command is split with shlex and executed
directly without a shell, so all control operators are denied and pass through
literally — pass a single concrete command in that mode.

Regardless of the mode, policy blocks: destructive commands (sudo, curl, wget,
ssh, scp, chmod 777, mkfs, dd, rm -rf); command substitution (backticks) and
brace expansion (unquoted { }); and redirects into files. Redirects are allowed
only to /dev/null and /dev/std{out,err} — to write or edit files, use
apply_patch for precise edits and write_file for deliberate full rewrites, not
shell redirection. Globbing (*, ?, [..]) reaches the program literally and is
expanded by tools that understand it (find, git).
run_shell is one-shot only: it blocks until the command exits, so don't use it
for background, long-running, or parallel processes — use run_shell_background
to start an HTTP server, dev server, or another process that must stay up while
you inspect it.
"""

RUN_SHELL_SPEC = ToolSpec(
    "run_shell",
    RUN_SHELL_DESCRIPTION,
    obj(
        {
            "command": strp(
                req=True,
                desc="Command with arguments, run from the workspace root. By default runs through /bin/sh -c, so && ; | and 2>&1 >/dev/null work; destructive commands and redirects into files are blocked by policy.",
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
