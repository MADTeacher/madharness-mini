"""JSON-RPC 2.0 helpers для минимального MCP lifecycle."""

from typing import Any

MCP_PROTOCOL_VERSION = "2025-11-25"


class JsonRpcBuilder:
    """Выдаёт монотонные id и собирает JSON-RPC сообщения для MCP."""

    def __init__(self):
        self._next_id = 1

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Собираем request, на который MCP-сервер обязан вернуть response."""

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        self._next_id += 1
        if params is not None:
            message["params"] = params
        return message

    def notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Собираем notification без id: ответа на него протокол не ждёт."""

        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        return message


def parse_response(message: dict[str, Any], expected_id: int) -> dict[str, Any]:
    """Достаём result из JSON-RPC response или превращаем error в RuntimeError."""

    if message.get("jsonrpc") != "2.0":
        raise RuntimeError("invalid JSON-RPC response: missing jsonrpc=2.0")
    if message.get("id") != expected_id:
        raise RuntimeError(
            f"invalid JSON-RPC response id: expected {expected_id}, got {message.get('id')}"
        )
    if "error" in message:
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            text = error.get("message", "unknown error")
            raise RuntimeError(f"MCP JSON-RPC error {code}: {text}")
        raise RuntimeError(f"MCP JSON-RPC error: {error}")
    if "result" not in message:
        raise RuntimeError("invalid JSON-RPC response: missing result")
    result = message["result"]
    if not isinstance(result, dict):
        raise RuntimeError("invalid JSON-RPC response: result must be object")
    return result


def method_not_found_response(request_id: Any, method: str) -> dict[str, Any]:
    """Отвечаем серверу, что client-side request в V1 не поддерживается."""

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }
