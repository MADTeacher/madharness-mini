"""Stdio transport для MCP-серверов на subprocess и JSON lines."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from typing import Any

from ..utils import clipped
from .config import McpServerConfig
from .protocol import (
    MCP_PROTOCOL_VERSION,
    JsonRpcBuilder,
    method_not_found_response,
    parse_response,
)

# Эти переменные обычно нужны локальным рантаймам вроде `npx`, но не раскрывают
# ключ LLM API и другие MADHARNESS_MINI_* настройки.
SAFE_INHERITED_ENV = {
    "ComSpec",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SystemRoot",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "WINDIR",
}


class StdioMcpClient:
    """Один запущенный MCP-сервер: initialize, tools/list и tools/call."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self.rpc = JsonRpcBuilder()
        self.process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self._stderr: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> list[dict[str, Any]]:
        """Запускаем сервер, проходим MCP lifecycle и возвращаем tools/list."""

        self.process = subprocess.Popen(
            [self.config.command, *self.config.args],
            cwd=self.config.cwd,
            env=_server_env(self.config.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name=f"mcp-{self.config.name}-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=f"mcp-{self.config.name}-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        init = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "madharness-mini",
                    "version": "0.1.0",
                },
            },
        )
        version = init.get("protocolVersion")
        if version != MCP_PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported MCP protocolVersion: {version}; expected {MCP_PROTOCOL_VERSION}"
            )
        self.notify("notifications/initialized", {})
        listed = self.request("tools/list", {})
        tools = listed.get("tools", [])
        if not isinstance(tools, list):
            raise RuntimeError("invalid MCP tools/list response: tools must be list")
        return tools

    def request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Отправляем JSON-RPC request и ждём response с нужным id."""

        message = self.rpc.request(method, params)
        expected_id = int(message["id"])
        self._send(message)
        deadline = time.monotonic() + self.config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"MCP request timed out: {method}; stderr: {self.stderr_excerpt()}"
                )
            try:
                incoming = self._messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if isinstance(incoming, Exception):
                raise incoming
            if _is_server_request(incoming):
                self._send(
                    method_not_found_response(
                        incoming.get("id"), str(incoming.get("method") or "")
                    )
                )
                continue
            if "id" not in incoming:
                continue
            if incoming.get("id") != expected_id:
                raise RuntimeError(
                    f"unexpected MCP response id: {incoming.get('id')}; expected {expected_id}"
                )
            return parse_response(incoming, expected_id)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Отправляем MCP notification без ожидания response."""

        self._send(self.rpc.notification(method, params))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Вызываем исходное имя MCP tool на сервере."""

        return self.request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

    def close(self) -> int | None:
        """Закрываем subprocess даже если сервер завис после закрытия stdin."""

        process = self.process
        if process is None:
            return None
        try:
            if process.poll() is not None:
                return process.returncode
            try:
                if process.stdin:
                    process.stdin.close()
            except OSError:
                pass
            try:
                return process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
            try:
                return process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait(timeout=2)
        finally:
            self._close_pipes()

    def stderr_excerpt(self, limit: int = 2000) -> str:
        """Коротко показываем stderr для диагностики, не раздувая trace."""

        return clipped("".join(self._stderr), limit).strip()

    def _send(self, message: dict[str, Any]) -> None:
        """Пишем одно JSON-RPC сообщение одной строкой в stdin сервера."""

        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError(f"MCP server {self.config.name} is not running")
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise RuntimeError(
                f"MCP server {self.config.name} stdin is closed; stderr: {self.stderr_excerpt()}"
            ) from exc

    def _read_stdout(self) -> None:
        """Читаем protocol stdout построчно и кладём JSON-сообщения в очередь."""

        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._messages.put(
                    RuntimeError(f"invalid MCP JSON from stdout: {exc}: {line[:200]}")
                )
                continue
            if not isinstance(message, dict):
                self._messages.put(RuntimeError("invalid MCP message: expected object"))
                continue
            self._messages.put(message)

    def _read_stderr(self) -> None:
        """Собираем stderr отдельно, чтобы он не смешивался с JSON-RPC."""

        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr.append(line)
            if sum(len(item) for item in self._stderr) > 6000:
                self._stderr = self._stderr[-20:]

    def _raise_if_process_exited(self) -> None:
        """Сообщаем о преждевременной смерти сервера до ответа."""

        process = self.process
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"MCP server {self.config.name} exited with code {process.returncode}; "
                f"stderr: {self.stderr_excerpt()}"
            )

    def _close_pipes(self) -> None:
        """Закрываем файловые объекты Popen, чтобы тесты не ловили warnings."""

        process = self.process
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except OSError:
                pass
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread:
                thread.join(timeout=0.2)


def _server_env(explicit: dict[str, str]) -> dict[str, str]:
    """Передаём серверу только безопасный минимум окружения и явный env."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key in SAFE_INHERITED_ENV and not key.startswith("MADHARNESS_MINI_")
    }
    env.update(explicit)
    return env


def _is_server_request(message: dict[str, Any]) -> bool:
    """Отличаем request сервера от response/notification в нашем V1 клиенте."""

    return "id" in message and "method" in message and "result" not in message
