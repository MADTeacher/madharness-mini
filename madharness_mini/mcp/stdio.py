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
    parse_response,
)

# Имена из окружения харнесса, которые `_server_env` копирует в subprocess
# MCP-сервера. PATH, TEMP и системные каталоги нужны локальным рантаймам
# вроде `npx`, чтобы сервер вообще стартовал.
SAFE_INHERITED_ENV = {
    "ComSpec",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SystemRoot",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "WINDIR",
}


def _server_env(explicit: dict[str, str]) -> dict[str, str]:
    """Передаем серверу только безопасный минимум окружения и явный env."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key in SAFE_INHERITED_ENV and not key.startswith("MADHARNESS_MINI_")
    }
    env.update(explicit)
    return env


def _is_server_request(message: dict[str, Any]) -> bool:
    """Детектор запрещенного в stateless MCP запроса сервера к клиенту."""

    return "id" in message and "method" in message and "result" not in message


class StdioMcpClient:
    """Клиент одного MCP-сервера, запускаемого в отдельном субпроцессе."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self.rpc = JsonRpcBuilder()
        self.process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self._stderr: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> list[dict[str, Any]]:
        """Запускаем сервер, проверяем server/discover и возвращаем список tools."""

        # Запускаем сервер в отдельном субпроцессе.
        self.process = subprocess.Popen(
            [self.config.command, *self.config.args],
            cwd=self.config.cwd,  # Рабочая директория сервера.
            env=_server_env(self.config.env),
            stdin=subprocess.PIPE,  # stdin сервера.
            stdout=subprocess.PIPE,  # stdout сервера.
            stderr=subprocess.PIPE,  # stderr сервера.
            text=True,  # Будем читать и писать строки, а не байты.
            encoding="utf-8",  # Кодировка для чтения и записи строк.
            errors="replace",  # Будем заменять невалидные символы на замещающие.
            bufsize=1,  # Будем читать и писать построчно, а не блоками.
        )
        # Запускаем по отдельному потоку для чтения stdout и stderr сервера.
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

        # Проверяем поддерживает ли сервер stateless MCP.
        # Современные серверы отвечают DiscoverResult,
        # legacy-серверы до initialize отвечают ошибкой или молчат.
        try:
            discover = self.request("server/discover")
        except RuntimeError as exc:
            raise RuntimeError(
                f"MCP server {self.config.name} failed server/discover "
                f"probe: {exc}; madharness-mini supports only stateless MCP"
                f" {MCP_PROTOCOL_VERSION} and cannot talk to servers "
                f"that require the legacy initialize handshake"
            ) from exc
        versions = discover.get("supportedVersions")
        if not isinstance(versions, list) or not all(
            isinstance(version, str) for version in versions
        ):
            raise RuntimeError(
                f"invalid MCP server/discover result from {self.config.name}: "
                "supportedVersions must be list of strings"
            )
        if MCP_PROTOCOL_VERSION not in versions:
            raise RuntimeError(
                f"MCP server {self.config.name} supports protocol versions "
                f"{versions}; madharness-mini requires "
                f"{MCP_PROTOCOL_VERSION}"
            )

        # Запрашиваем список tools и проверяем и проверяем,
        # что сервер вернул то, что мы ожидаем.
        listed = self.request("tools/list", {})
        tools = listed.get("tools", [])
        if not isinstance(tools, list):
            raise RuntimeError(
                "invalid MCP tools/list response: tools must be list",
            )
        return tools

    def request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Отправляем JSON-RPC request и ждем response с тем же id.

        Запрос пишется в stdin субпроцесса. Ответ забирается из очереди,
        которую наполняет поток чтения stdout. Ожидание ограничено
        timeout_seconds и идет короткими интервалами, чтобы заметить
        завершение процесса до конца таймаута.
        """

        message = self.rpc.request(method, params)
        expected_id = int(message["id"])
        self._send(message)
        # Момент, после которого ожидание запроса завершается по таймауту.
        deadline = time.monotonic() + self.config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"MCP request timed out: {method}; "
                    f"stderr: {self.stderr_excerpt()}"
                )
            try:
                # Опрос очереди не дольше 0.1 с. Между опросами проверяем,
                # не завершился ли субпроцесс: блокировка на весь
                # timeout_seconds отложила бы проверку до дедлайна.
                incoming = self._messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            # Поток чтения stdout кладет в очередь и сообщения, и Exception
            # (битый JSON и другие ошибки разбора). Exception пробрасываем.
            if isinstance(incoming, Exception):
                raise incoming
            if _is_server_request(incoming):
                # Stateless MCP запрещает серверу JSON-RPC request по stdio.
                # Вместо него — result типа input_required. Сообщение с id
                # и method без result — ошибка протокола.
                raise RuntimeError(
                    f"invalid MCP message from {self.config.name}: "
                    f"server sent request {incoming.get('method')!r}; "
                    "stateless MCP servers must not send requests"
                )
            # Notification без id к этому request не относится.
            if "id" not in incoming:
                continue
            # Ждем ответ только на текущий запрос. Другого запроса в ожидании
            # нет, поэтому сообщение с чужим id — ошибка протокола.
            if incoming.get("id") != expected_id:
                raise RuntimeError(
                    f"unexpected MCP response id: {incoming.get('id')}; "
                    f"expected {expected_id}"
                )
            # parse_response проверяет форму ответа и при наличии error
            # выбрасывает RuntimeError.
            return parse_response(incoming, expected_id)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Вызываем tools/call: имя tool уходит на сервер как есть."""

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
            # Если процесс уже завершен, возвращаем его код подтверждения
            if process.poll() is not None:
                return process.returncode
            # Закрываем stdin сервера, после чего тот сам должен
            # запустить процесс завершения своей работы.
            # Перехваченное исключение OSError означает, что stdin уже закрыт
            try:
                if process.stdin:
                    process.stdin.close()
            except OSError:
                pass
            # Даем серверу 2 секунды на завершение своей работы
            try:
                return process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Если сервер не завершился за 2 секунды, отправляем ему
                # сигнал SIGTERM (или TerminateProcess на Windows)
                process.terminate()
            # Даем серверу 2 секунды на завершение после terminate
            try:
                return process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Если сервер не завершился за 2 секунды после terminate,
                # отправляем ему сигнал SIGKILL. На Windows kill совпадает
                # с terminate.
                process.kill()
                return process.wait(timeout=2) # снова ждем
        finally:
            # Закрываем каналы и ждем завершения потоков чтения
            self._close_pipes()

    def stderr_excerpt(self, limit: int = 2000) -> str:
        """Коротко показываем stderr для диагностики, не раздувая trace."""

        return clipped("".join(self._stderr), limit).strip()

    def _send(self, message: dict[str, Any]) -> None:
        """Пишем одно JSON-RPC сообщение одной строкой в stdin сервера."""

        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError(
                f"MCP server {self.config.name} is not running",
            )
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise RuntimeError(
                f"MCP server {self.config.name} stdin is closed; "
                f"stderr: {self.stderr_excerpt()}"
            ) from exc

    def _read_stdout(self) -> None:
        """Читаем stdout построчно. В очередь кладем JSON-объект или ошибку разбора."""

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
                self._messages.put(
                    RuntimeError("invalid MCP message: expected object"),
                )
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
                f"MCP server {self.config.name} exited with code "
                f"{process.returncode}; stderr: {self.stderr_excerpt()}"
            )

    def _close_pipes(self) -> None:
        """Закрываем stdin, stdout и stderr процесса и ждем завершения потоков чтения."""

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
