"""Фоновый запуск shell-команды: HTTP-сервер, dev-сервер, watcher.

В отличие от run_shell, этот инструмент не ждёт завершения команды. Он держит
процесс в пуле provider'а и даёт модели три действия через один tool: start
(поднять), status (прочитать накопленный вывод и состояние), stop (остановить).
Provider реализует close(trace): при завершении сессии ToolRegistry.close()
гарантированно гасит все висящие процессы, как это делает McpToolProvider.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from ..utils import clipped, fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec

TOOL_NAME = "run_shell_background"

RUN_SHELL_BACKGROUND_DESCRIPTION = """Start and manage a background process in the workspace.

Use this for a server, dev build, or watcher that must stay running while you
inspect it: a local HTTP server to open a static page in a browser, a Vite/webpack
dev server, a file watcher, etc. run_shell blocks until the command exits, which
is wrong for these cases — this tool returns immediately.

One tool, three actions:

- action="start": launch command (same policy as run_shell — destructive commands
  and redirects into files are blocked). Returns an id (for later status/stop
  calls) and pid. The process runs from cwd inside the workspace.
- action="status": read accumulated stdout/stderr and the current running/exit
  state for a previously started id. Output is clipped; call it again later to
  see fresh output appended since start.
- action="stop": terminate the process (SIGTERM, then SIGKILL after a grace
  period) and report its exit code. Always stop servers when done so ports and
  resources are freed.

All background processes are also stopped automatically when the session ends.
Do not rely on the auto-cleanup for correctness though — stop explicitly so you
can read the final exit code and confirm the server came down cleanly. The
number of simultaneously running background processes is capped by
max_background_processes.
"""


class ShellBackgroundProvider:
    """Держит пул фоновых процессов; cleanup в ToolRegistry.close().

    Паттерн как у McpToolProvider: процессы хранятся в инстансе provider'а
    (не в ToolContext), а handler замыкается на self. Registry вызывает
    close(trace) в finally-блоке loop.run_agent, поэтому процессы гасятся и при
    нормальном завершении, и при ошибке/прерывании.
    """

    def __init__(self) -> None:
        # id -> запись о процессе. id выдаём последовательный (bg1, bg2, ...).
        self._processes: dict[str, _BgProcess] = {}
        self._counter = 0

    def specs(self, ctx: ToolContext) -> list[ToolSpec]:
        """Отдаём единственный tool; handler замыкается на этот provider."""

        provider = self

        def handler(tool_ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
            return _handle(provider, tool_ctx, args)

        return [
            ToolSpec(
                name=TOOL_NAME,
                description=RUN_SHELL_BACKGROUND_DESCRIPTION,
                parameters=obj(
                    {
                        "action": strp(
                            req=True,
                            desc='"start" to launch command, "status" to read output/state, "stop" to terminate',
                        ),
                        "command": strp(
                            desc='For action="start": the command to run in the background (same policy as run_shell).'
                        ),
                        "cwd": strp(
                            ".",
                            'For action="start": workspace-relative directory to run from.',
                        ),
                        "id": strp(
                            desc='For action="status"/"stop": the id returned by a previous start.',
                        ),
                    },
                    ["action"],
                ),
                handler=handler,
            )
        ]

    def close(self, trace: Any = None) -> None:
        """Гасим все висящие процессы при завершении сессии.

        Эскалация как у StdioMcpClient.close: SIGTERM → grace → SIGKILL. Ошибки
        одного процесса не должны ронять очистку остальных, поэтому ловим и
        пишем в trace, если он есть.
        """

        for bg_id in list(self._processes.keys()):
            exit_code = self._terminate(bg_id)
            if trace is not None:
                try:
                    trace.write(
                        "shell_background_stopped",
                        tool=TOOL_NAME,
                        id=bg_id,
                        exit_code=exit_code,
                        reason="session_end",
                    )
                except Exception:
                    pass

    def _terminate(self, bg_id: str) -> int | None:
        """Останавливаем один процесс эскалацией SIGTERM → SIGKILL.

        Возвращает exit_code или None, если процесс уже завершился. Удаляет
        запись из пула. Используем process group (start_new_session=True при
        старте), чтобы убить и дочерние процессы команды.
        """

        bg = self._processes.pop(bg_id, None)
        if bg is None:
            return None
        return _stop_process(bg.process)


class _BgProcess:
    """Процесс + неблокируемо накопленный stdout/stderr.

    Читаем pipe'ы неблокирующе в момент status, чтобы не держать daemon-тред
    (проще для учебного harness) и не плодить гонки при cleanup.
    """

    __slots__ = ("process", "stdout_buf", "stderr_buf")

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []

    def drain(self) -> None:
        """Забираем доступные данные из pipe'ов без блокировки."""

        self._drain_stream(self.process.stdout, self.stdout_buf)
        self._drain_stream(self.process.stderr, self.stderr_buf)

    @staticmethod
    def _drain_stream(stream: Any, buf: list[str]) -> None:
        # Без доступного потока читать нельзя — guard через fileno().
        if stream is None:
            return
        try:
            import select

            while True:
                ready, _, _ = select.select([stream], [], [], 0)
                if not ready:
                    return
                chunk = stream.read(4096)
                if not chunk:
                    return
                buf.append(chunk)
        except Exception:
            # select/read могут падать на закрытом потоке — просто прекращаем.
            return


def _handle(
    provider: ShellBackgroundProvider, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    """Маршрутизируем action в start/status/stop."""

    action = str(args.get("action") or "").strip().lower()
    if action == "start":
        return _start(provider, ctx, args)
    if action == "status":
        return _status(provider, ctx, args)
    if action == "stop":
        return _stop(provider, ctx, args)
    return fail(TOOL_NAME, f"unknown action: {action!r}; expected start/status/stop")


def _start(
    provider: ShellBackgroundProvider, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    """Поднимаем процесс после проверки политики и лимита процессов."""

    command = str(args.get("command") or "").strip()
    if not command:
        return fail(TOOL_NAME, "command is required for action=start")
    allowed, reason = ctx.policy.shell_allowed(command)
    if not allowed:
        return fail(TOOL_NAME, reason, command=command)
    cwd_raw = str(args.get("cwd") or ".")
    cwd, err = ctx.policy.safe_path(cwd_raw)
    if err or not cwd:
        return fail(TOOL_NAME, err or f"invalid cwd: {cwd_raw}", command=command)
    if not cwd.is_dir():
        return fail(TOOL_NAME, f"cwd is not a directory: {cwd_raw}", command=command)

    max_procs = int(ctx.cfg.data.get("max_background_processes", 4))
    running = sum(1 for bg in provider._processes.values() if bg.process.poll() is None)
    if running >= max_procs:
        return fail(
            TOOL_NAME,
            f"max_background_processes limit reached ({max_procs}); stop an existing process first",
            command=command,
        )

    # start_new_session=True создаёт новую process group — потом убиваем её
    # целиком через os.killpg, чтобы не осталось дочерних процессов сервера.
    use_shell = bool(ctx.cfg.data.get("allow_shell_interpreter", True))
    try:
        if use_shell:
            process = subprocess.Popen(
                command,
                shell=True,
                executable="/bin/sh",
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        else:
            import shlex

            process = subprocess.Popen(
                shlex.split(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
    except OSError as exc:
        return fail(TOOL_NAME, f"failed to start: {exc}", command=command)

    provider._counter += 1
    bg_id = f"bg{provider._counter}"
    provider._processes[bg_id] = _BgProcess(process)

    try:
        cwd_display = str(cwd.relative_to(ctx.cfg.root))
    except ValueError:
        cwd_display = str(cwd)

    if ctx.trace is not None:
        ctx.trace.write(
            "shell_background_started",
            tool=TOOL_NAME,
            id=bg_id,
            pid=process.pid,
            command=command,
            cwd=cwd_display or ".",
        )
    return ok(
        TOOL_NAME,
        f"started {bg_id} (pid {process.pid})",
        id=bg_id,
        pid=process.pid,
        command=command,
        cwd=cwd_display or ".",
    )


def _status(
    provider: ShellBackgroundProvider, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    """Читаем накопленный вывод и состояние процесса по id."""

    bg_id = str(args.get("id") or "").strip()
    bg = provider._processes.get(bg_id)
    if bg is None:
        return fail(TOOL_NAME, f"unknown id: {bg_id!r}")
    bg.drain()
    returncode = bg.process.poll()
    running = returncode is None
    return ok(
        TOOL_NAME,
        f"{'running' if running else 'exited'} ({bg_id})",
        id=bg_id,
        running=running,
        returncode=returncode,
        stdout=clipped("".join(bg.stdout_buf)),
        stderr=clipped("".join(bg.stderr_buf)),
    )


def _stop(
    provider: ShellBackgroundProvider, ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    """Останавливаем процесс и фиксируем exit_code в trace."""

    bg_id = str(args.get("id") or "").strip()
    bg = provider._processes.get(bg_id)
    if bg is None:
        return fail(TOOL_NAME, f"unknown id: {bg_id!r}")
    # Дочитываем хвост вывода перед смертью, чтобы модель видела финальные строки.
    bg.drain()
    exit_code = _stop_process(bg.process)
    stdout_tail = "".join(bg.stdout_buf)
    stderr_tail = "".join(bg.stderr_buf)
    provider._processes.pop(bg_id, None)
    if ctx.trace is not None:
        ctx.trace.write(
            "shell_background_stopped",
            tool=TOOL_NAME,
            id=bg_id,
            exit_code=exit_code,
            reason="explicit_stop",
        )
    return ok(
        TOOL_NAME,
        f"stopped {bg_id} (exit {exit_code})",
        id=bg_id,
        exit_code=exit_code,
        stdout=clipped(stdout_tail),
        stderr=clipped(stderr_tail),
    )


def _stop_process(process: subprocess.Popen) -> int | None:
    """Эскалация остановки: SIGTERM группе → ждём → SIGKILL группе.

    Используем process group (pgid = pid при start_new_session=True), чтобы
    убить и команду, и её дочерние процессы. Возвращает exit_code или None,
    если процесс уже завершился до вызова. В конце закрывает pipe'ы — иначе
    TextIOWrapper stdout/stderr остаётся незакрытым и Python ругается
    ResourceWarning при сборке мусора.
    """

    def _close_pipes() -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    try:
        if process.poll() is not None:
            return process.returncode
        pgid = os.getpgid(process.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return process.returncode
        try:
            return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return None
    finally:
        _close_pipes()
