"""Основной анализ тепловой карты по нормализованным событиям."""

from __future__ import annotations

from collections import Counter, deque

from .aggregate import aggregate_turns, session_report
from .features import detect_cold_gaps, score_fragment
from .fragments import extract_fragments
from .packets import reconstruct_packets
from .schema import AnalysisResult, FragmentHeatRecord, SessionEvent


def analyze_events(
    events: list[SessionEvent],
    warnings: list[dict] | None = None,
) -> AnalysisResult:
    """Считает fragments, packets, heat, findings и итоговый отчет."""

    base_warnings = list(warnings or [])
    fragments = extract_fragments(events)
    fragment_by_id = {fragment.fragment_id: fragment for fragment in fragments}
    packets, packet_warnings = reconstruct_packets(events)
    all_warnings = [*base_warnings, *packet_warnings]
    fragment_heat: list[FragmentHeatRecord] = []
    recent: deque[list[str]] = deque(maxlen=5)
    recent_counts: Counter[str] = Counter()
    previous_input_tokens = 0

    for packet in packets:
        active_ids = [item.fragment_id for item in packet.fragments]
        active_hash_counts = Counter(
            fragment_by_id[item].content_hash
            for item in active_ids
            if item in fragment_by_id and fragment_by_id[item].content_hash
        )
        growth_slope = (
            max(packet.input_tokens - previous_input_tokens, 0)
            / max(packet.input_tokens, 1)
            if previous_input_tokens
            else 0.0
        )
        previous_input_tokens = packet.input_tokens
        for fragment_id in active_ids:
            fragment = fragment_by_id.get(fragment_id)
            if not fragment:
                continue
            axes, heat, reasons, confidence = score_fragment(
                fragment,
                packet,
                active_hash_counts,
                recent_counts,
                growth_slope,
            )
            fragment_heat.append(
                FragmentHeatRecord(
                    session_id=packet.session_id,
                    model_call_id=packet.model_call_id,
                    fragment_id=fragment.fragment_id,
                    heat=round(heat, 4),
                    confidence=round(confidence, 4),
                    axes={key: round(value, 4) for key, value in axes.items()},
                    reasons=reasons,
                    evidence_event_ids=[fragment.created_by_event_id]
                    if fragment.created_by_event_id
                    else [],
                )
            )
        recent.append(active_ids)
        recent_counts = Counter(item for packet_ids in recent for item in packet_ids)

    findings = detect_cold_gaps(events)
    turn_heat = aggregate_turns(packets, fragment_heat, findings)
    session_id = events[0].session_id if events else "unknown-session"
    report = session_report(session_id, packets, turn_heat, findings, all_warnings)
    return AnalysisResult(
        session_id=session_id,
        events=events,
        fragments=fragments,
        packets=packets,
        fragment_heat=fragment_heat,
        turn_heat=turn_heat,
        findings=findings,
        warnings=all_warnings,
        session_report=report,
    )
