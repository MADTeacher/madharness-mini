"""Выделение фрагментов контекста из нормализованных событий."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .privacy import contains_secret, redact_text
from .schema import ContextFragmentRecord, SessionEvent


def extract_fragments(events: list[SessionEvent]) -> list[ContextFragmentRecord]:
    """Создает фрагменты из context_packet и legacy context_report."""

    fragments: dict[str, ContextFragmentRecord] = {}
    for event in events:
        if event.event_type == "model_call":
            for fragment in _fragments_from_model_call(event):
                fragments.setdefault(fragment.fragment_id, fragment)
        if event.event_type == "tool_result":
            fragment = _fragment_from_tool_result(event)
            if fragment:
                fragments.setdefault(fragment.fragment_id, fragment)
    return list(fragments.values())


def fragment_id_for_unit(session_id: str, unit: dict[str, Any]) -> str:
    """Делаем стабильный id по source и hash, чтобы повторы были видны."""

    source_type = str(unit.get("source_type") or "unknown")
    source_ref = str(unit.get("source_ref") or unit.get("unit_id") or "unit")
    digest = str(unit.get("content_hash") or _hash_payload(unit))
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in source_ref)[:48]
    return f"frag-{source_type}-{cleaned}-{digest[:12]}"


def _fragments_from_model_call(event: SessionEvent) -> list[ContextFragmentRecord]:
    report = event.payload.get("context_report")
    if not isinstance(report, dict):
        return []
    packet = report.get("context_packet")
    if isinstance(packet, dict) and isinstance(packet.get("units"), list):
        return [_fragment_from_unit(event, unit) for unit in packet["units"]]
    return _legacy_fragments(event, report)


def _fragment_from_unit(
    event: SessionEvent,
    unit: dict[str, Any],
) -> ContextFragmentRecord:
    source_type = str(unit.get("source_type") or "unknown")
    source_name = str(unit.get("source_name") or "")
    metadata = dict(unit.get("metadata") or {})
    return ContextFragmentRecord(
        fragment_id=fragment_id_for_unit(event.session_id, unit),
        session_id=event.session_id,
        source_type=source_type,
        source_name=source_name,
        tokens=int(unit.get("tokens_estimate") or 0),
        token_count_method="char_estimate",
        trust=_trust_for(source_type),
        taint="unknown" if source_type in {"tool_output", "unknown"} else "none",
        validity="unknown",
        target_paths=_target_paths_from_metadata(metadata),
        created_by_event_id=event.event_id,
        content_hash=str(unit.get("content_hash") or ""),
        content_excerpt_redacted="",
        metadata={
            "source_ref": str(unit.get("source_ref") or ""),
            "included_because": str(unit.get("included_because") or ""),
            "confidence": float(unit.get("confidence") or 0.0),
            **metadata,
        },
    )


def _legacy_fragments(
    event: SessionEvent,
    report: dict[str, Any],
) -> list[ContextFragmentRecord]:
    fragments: list[ContextFragmentRecord] = []
    for item in report.get("fragments") or []:
        if not isinstance(item, dict):
            continue
        fragment_id = f"legacy-fragment-{item.get('id') or len(fragments)}"
        fragments.append(
            ContextFragmentRecord(
                fragment_id=fragment_id,
                session_id=event.session_id,
                source_type="context_fragment",
                source_name=str(item.get("source") or ""),
                tokens=max(int(item.get("chars") or 0) // 3, 1),
                token_count_method="char_estimate",
                trust="unknown",
                created_by_event_id=event.event_id,
                content_hash=_hash_payload(item),
                metadata={"legacy": True, "confidence": 0.55},
            )
        )
    history = report.get("history")
    if isinstance(history, dict):
        for item in history.get("included_entries") or []:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index") or 0)
            fragments.append(
                ContextFragmentRecord(
                    fragment_id=f"legacy-history-{index}",
                    session_id=event.session_id,
                    source_type="assistant_message"
                    if item.get("kind") == "assistant"
                    else "tool_output",
                    source_name=str(item.get("kind") or "history"),
                    tokens=int(item.get("tokens_estimate") or 0),
                    token_count_method="char_estimate",
                    trust="unknown",
                    taint="unknown",
                    created_by_event_id=event.event_id,
                    content_hash=_hash_payload(item),
                    metadata={"legacy": True, "confidence": 0.55},
                )
            )
    return fragments


def _fragment_from_tool_result(event: SessionEvent) -> ContextFragmentRecord | None:
    tool = str(event.payload.get("tool") or "")
    observation = event.payload.get("observation")
    if not isinstance(observation, dict):
        return None
    text = json.dumps(observation, ensure_ascii=False, sort_keys=True)
    excerpt, secret = redact_text(text)
    return ContextFragmentRecord(
        fragment_id=f"event-tool-{event.event_id.split(':')[-1]}",
        session_id=event.session_id,
        source_type=_tool_source_type(tool, observation),
        source_name=tool,
        tokens=max(len(text.encode("utf-8")) // 3, 1),
        token_count_method="char_estimate",
        trust="local_verified",
        taint="secret" if secret or contains_secret(text) else "none",
        validity="unknown",
        target_paths=_target_paths_from_tool(event.payload),
        created_by_event_id=event.event_id,
        content_hash=_hash_payload(observation),
        content_excerpt_redacted=excerpt,
        metadata={"included_in_prompt": False},
    )


def _tool_source_type(tool: str, observation: dict[str, Any]) -> str:
    """Классифицируем tool output для осей heat."""

    if tool == "read_file":
        return "file_snippet"
    if tool == "run_shell" and _looks_like_test(observation):
        return "test_result"
    return "tool_output"


def _looks_like_test(observation: dict[str, Any]) -> bool:
    text = " ".join(str(observation.get(key) or "") for key in ("command", "summary"))
    return any(marker in text.lower() for marker in ("test", "pytest", "unittest"))


def _target_paths_from_metadata(metadata: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("path", "target_path", "target_paths"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
        if isinstance(value, list):
            paths.extend(str(item) for item in value if str(item))
    return sorted(set(paths))


def _target_paths_from_tool(payload: dict[str, Any]) -> list[str]:
    args = payload.get("args")
    if not isinstance(args, dict):
        return []
    paths = []
    for key in ("path", "cwd"):
        if isinstance(args.get(key), str):
            paths.append(args[key])
    return sorted(set(paths))


def _trust_for(source_type: str) -> str:
    if source_type in {"system_instruction", "developer_instruction", "user_message"}:
        return "trusted"
    if source_type in {"file_snippet", "test_result"}:
        return "local_verified"
    if source_type == "tool_output":
        return "local_verified"
    if source_type == "assistant_message":
        return "generated"
    return "unknown"


def _hash_payload(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
