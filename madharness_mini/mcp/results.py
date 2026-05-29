"""Преобразование MCP `tools/call` result в observation харнесса."""

import json
from typing import Any

from ..utils import clipped, fail, ok


def mcp_result_to_observation(
    exported_name: str,
    server_name: str,
    tool_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Сообщаем модели результат MCP tool в привычном формате ok/fail."""

    lines: list[str] = []
    images: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    unknown: list[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            unknown.append(f"non-object content: {type(item).__name__}")
            continue
        kind = item.get("type")
        if kind == "text":
            text = item.get("text", "")
            lines.append(text if isinstance(text, str) else json.dumps(text))
        elif kind in {"image", "audio"}:
            images.append(_media_metadata(item, kind))
        elif kind in {"resource", "resource_link"}:
            resources.append(_resource_metadata(item))
        else:
            unknown.append(f"unsupported MCP content type: {kind}")

    content = clipped("\n".join(line for line in lines if line))
    data: dict[str, Any] = {}
    if content:
        data["content"] = content
    structured = result.get("structuredContent")
    if structured is not None:
        data["data"] = structured
    if images:
        data["images"] = images
    if resources:
        data["resources"] = resources
    if unknown:
        data["diagnostics"] = unknown

    summary = f"MCP tool {server_name}.{tool_name} completed"
    if result.get("isError") is True:
        return fail(exported_name, summary, **data)
    return ok(exported_name, summary, **data)


def _media_metadata(item: dict[str, Any], kind: str) -> dict[str, Any]:
    """Оставляем метаданные media content, не отправляя base64 модели."""

    raw_data = item.get("data")
    return {
        "type": kind,
        "mime_type": item.get("mimeType") or item.get("mime_type") or "",
        "base64_chars": len(raw_data) if isinstance(raw_data, str) else 0,
    }


def _resource_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Коротко описываем resource content без дополнительной загрузки."""

    resource = item.get("resource")
    if not isinstance(resource, dict):
        resource = item
    return {
        "uri": resource.get("uri", ""),
        "name": resource.get("name", ""),
        "mime_type": resource.get("mimeType") or resource.get("mime_type") or "",
        "description": resource.get("description", ""),
    }
