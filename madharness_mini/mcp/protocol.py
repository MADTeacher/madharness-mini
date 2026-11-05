"""JSON-RPC 2.0 helpers для stateless MCP (спецификация 2026-07-28)."""

from typing import Any

MCP_PROTOCOL_VERSION = "2026-07-28"

CLIENT_INFO = {"name": "madharness-mini", "version": "0.1.0"}

# Известные типы результатов, которые может возвращать сервер
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
    """Выдает id и собирает JSON-RPC сообщения для MCP."""

    def __init__(self):
        # инициализируем счетчик id
        self._next_id = 1

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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


def parse_response(
    message: dict[str, Any],
    expected_id: int,
) -> dict[str, Any]:
    """Достаем result из JSON-RPC response или превращаем error в RuntimeError."""

    # проверяем версию jsonrpc
    if message.get("jsonrpc") != "2.0":
        raise RuntimeError("invalid JSON-RPC response: missing jsonrpc=2.0")

    # проверяем id сообщения
    if message.get("id") != expected_id:
        raise RuntimeError(
            f"invalid JSON-RPC response id: expected {expected_id}, "
            f"got {message.get('id')}"
        )

    # проверяем наличие error
    if "error" in message:
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            text = error.get("message", "unknown error")
            raise RuntimeError(f"MCP JSON-RPC error {code}: {text}")
        raise RuntimeError(f"MCP JSON-RPC error: {error}")

    # проверяем наличие result
    if "result" not in message:
        raise RuntimeError("invalid JSON-RPC response: missing result")

    # получаем result и проверяем его тип
    result = message["result"]
    if not isinstance(result, dict):
        raise RuntimeError(
            "invalid JSON-RPC response: result must be object",
        )
    # Отсутствие resultType допускается только для серверов старых ревизий.
    # Клиент обязан трактовать его как complete.
    result_type = result.get("resultType", "complete")
    if result_type not in KNOWN_RESULT_TYPES:
        raise RuntimeError(
            f"invalid MCP result: unknown resultType: {result_type}",
        )
    return result
