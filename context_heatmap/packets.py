"""Восстановление prompt-пакетов из нормализованных событий."""

from __future__ import annotations

from typing import Any

from .fragments import _classification_for_source_type, fragment_id_for_unit
from .schema import ContextPacketRecord, PacketFragment, SessionEvent


def reconstruct_packets(events: list[SessionEvent]) -> tuple[list[ContextPacketRecord], list[dict[str, Any]]]:
    """Строит ContextPacket для каждого model_call."""

    packets: list[ContextPacketRecord] = []
    warnings: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "model_call":
            continue
        packet, packet_warnings = _packet_from_event(event)
        packets.append(packet)
        warnings.extend(packet_warnings)
    return packets, warnings


def _packet_from_event(
    event: SessionEvent,
) -> tuple[ContextPacketRecord, list[dict[str, Any]]]:
    payload = event.payload
    report = payload.get("context_report")
    model_call_id = str(payload.get("model_call_id") or f"{event.session_id}:{event.turn_id}")
    if isinstance(report, dict):
        context_packet = report.get("context_packet")
        if isinstance(context_packet, dict) and isinstance(context_packet.get("units"), list):
            return _packet_from_context_packet(event, model_call_id, report, context_packet)
        return _legacy_packet(event, model_call_id, report)
    warning = _warning(event, "legacy_trace_without_context_report")
    return (
        ContextPacketRecord(
            model_call_id=model_call_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            input_tokens=0,
            context_window_tokens=0,
            fragments=[],
            reconstruction_confidence=0.35,
            warnings=[warning["kind"]],
        ),
        [warning],
    )


def _packet_from_context_packet(
    event: SessionEvent,
    model_call_id: str,
    report: dict[str, Any],
    context_packet: dict[str, Any],
) -> tuple[ContextPacketRecord, list[dict[str, Any]]]:
    fragments: list[PacketFragment] = []
    for unit in context_packet.get("units") or []:
        if not isinstance(unit, dict):
            continue
        fragments.append(
            _packet_fragment_from_unit(event, unit)
        )
    warnings = list(context_packet.get("warnings") or [])
    return (
        ContextPacketRecord(
            model_call_id=model_call_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            input_tokens=int(
                context_packet.get("request_tokens_estimate")
                or report.get("request_tokens_estimate")
                or 0
            ),
            context_window_tokens=int(report.get("max_tokens") or 0),
            fragments=fragments,
            reconstruction_confidence=0.9,
            warnings=[str(item) for item in warnings],
        ),
        [],
    )


def _packet_fragment_from_unit(
    event: SessionEvent,
    unit: dict[str, Any],
) -> PacketFragment:
    """Строим позицию prompt unit вместе с классификацией слоя."""

    source_type = str(unit.get("source_type") or "unknown")
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    classification = _classification_for_source_type(source_type)
    for key, fallback in list(classification.items()):
        value = unit.get(key)
        if value is None:
            value = metadata.get(key)
        classification[key] = str(value or fallback)
    return PacketFragment(
        fragment_id=fragment_id_for_unit(event.session_id, unit),
        position_start=int(unit.get("position_start") or 0),
        position_end=int(unit.get("position_end") or 0),
        tokens=int(unit.get("tokens_estimate") or 0),
        source_type=source_type,
        **classification,
    )


def _legacy_packet(
    event: SessionEvent,
    model_call_id: str,
    report: dict[str, Any],
) -> tuple[ContextPacketRecord, list[dict[str, Any]]]:
    fragments: list[PacketFragment] = []
    cursor = 0
    for item in report.get("fragments") or []:
        if not isinstance(item, dict):
            continue
        tokens = max(int(item.get("chars") or 0) // 3, 1)
        fragment_id = f"legacy-fragment-{item.get('id') or len(fragments)}"
        fragments.append(
            PacketFragment(
                fragment_id=fragment_id,
                position_start=cursor,
                position_end=cursor + tokens,
                tokens=tokens,
                source_type="context_fragment",
                **_classification_for_source_type("context_fragment"),
            )
        )
        cursor += tokens
    history = report.get("history")
    if isinstance(history, dict):
        for item in history.get("included_entries") or []:
            if not isinstance(item, dict):
                continue
            tokens = int(item.get("tokens_estimate") or 0)
            index = int(item.get("index") or 0)
            source_type = "assistant_message" if item.get("kind") == "assistant" else "tool_output"
            fragments.append(
                PacketFragment(
                    fragment_id=f"legacy-history-{index}",
                    position_start=cursor,
                    position_end=cursor + tokens,
                    tokens=tokens,
                    source_type=source_type,
                    **_classification_for_source_type(source_type),
                )
            )
            cursor += tokens
    warning = _warning(event, "legacy_trace_without_context_packet")
    return (
        ContextPacketRecord(
            model_call_id=model_call_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            input_tokens=int(report.get("request_tokens_estimate") or cursor),
            context_window_tokens=int(report.get("max_tokens") or 0),
            fragments=fragments,
            reconstruction_confidence=0.55,
            warnings=[warning["kind"]],
        ),
        [warning],
    )


def _warning(event: SessionEvent, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "event_id": event.event_id,
    }
