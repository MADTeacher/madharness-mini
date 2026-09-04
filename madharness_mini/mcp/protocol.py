"""JSON-RPC 2.0 helpers для stateless MCP (спецификация 2026-07-28)."""

from typing import Any

MCP_PROTOCOL_VERSION = "2026-07-28"

CLIENT_INFO = {"name": "madharness-mini", "version": "0.1.0"}

# Ядро спецификации определяет complete и input_required; расширения могут
# добавлять свои значения, но харнесс не объявляет поддержку расширений.
KNOWN_RESULT_TYPES = {"complete", "input_required"}


def request_meta() -> dict[str, Any]:
    """Собираем обязательный `_meta` каждого запроса stateless MCP.

    Протокол больше не имеет initialize-handshake, поэтому версия протокола,
    возможности и имя клиента передаются в `_meta` каждого запроса.
    """

    return {
        "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": dict(CLIENT_INFO),
        "io.modelcontextprotocol/clientCapabilities": {},
    }


class JsonRpcBuilder:
    """Выдаёт монотонные id и собирает JSON-RPC сообщения для MCP."""

    def __init__(self):
        self._next_id = 1

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Собираем request с обязательным `_meta`, на который сервер обязан ответить."""

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        self._next_id += 1
        merged = dict(params) if params is not None else {}
        merged["_meta"] = request_meta()
        message["params"] = merged
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
    # Отсутствие resultType допускается только для серверов старых ревизий;
    # клиент обязан трактовать его как complete.
    result_type = result.get("resultType", "complete")
    if result_type not in KNOWN_RESULT_TYPES:
        raise RuntimeError(f"invalid MCP result: unknown resultType: {result_type}")
    return result
