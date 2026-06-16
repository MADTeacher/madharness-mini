"""Агрегация fragment heat в показатели turn/session."""

from __future__ import annotations

from collections import Counter

from .schema import ContextPacketRecord, FragmentHeatRecord, Finding, TurnHeatRecord


def aggregate_turns(
    packets: list[ContextPacketRecord],
    fragment_heat: list[FragmentHeatRecord],
    findings: list[Finding],
) -> list[TurnHeatRecord]:
    """Считает агрегаты по каждому обращению к модели."""

    heat_by_call: dict[str, list[FragmentHeatRecord]] = {}
    cold_by_turn: dict[int, float] = {}
    for heat in fragment_heat:
        heat_by_call.setdefault(heat.model_call_id, []).append(heat)
    for finding in findings:
        if finding.kind == "cold_gap":
            cold_by_turn[finding.turn_id] = max(
                cold_by_turn.get(finding.turn_id, 0.0),
                float(finding.scores.get("cold_gap_score") or finding.confidence),
            )

    result: list[TurnHeatRecord] = []
    previous_tokens = 0
    for packet in packets:
        records = heat_by_call.get(packet.model_call_id, [])
        token_by_fragment = {item.fragment_id: item.tokens for item in packet.fragments}
        total_tokens = max(sum(token_by_fragment.values()), 1)
        red_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if item.heat >= 0.75
        )
        stale_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if float(item.axes.get("staleness") or 0.0) >= 0.50
        )
        raw_tool_tokens = sum(
            fragment.tokens
            for fragment in packet.fragments
            if fragment.source_type == "tool_output"
        )
        evidence_tokens = sum(
            fragment.tokens
            for fragment in packet.fragments
            if fragment.source_type
            in {"user_message", "file_snippet", "test_result", "tool_output"}
        )
        tainted_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if float(item.axes.get("taint") or 0.0) >= 0.50
        )
        reason_counts = Counter(
            reason for item in records for reason in item.reasons
        )
        growth_slope = (
            max(packet.input_tokens - previous_tokens, 0) / max(packet.input_tokens, 1)
            if previous_tokens
            else 0.0
        )
        previous_tokens = packet.input_tokens
        result.append(
            TurnHeatRecord(
                session_id=packet.session_id,
                model_call_id=packet.model_call_id,
                turn_id=packet.turn_id,
                red_token_share=round(red_tokens / total_tokens, 4),
                stale_token_share=round(stale_tokens / total_tokens, 4),
                raw_tool_share=round(raw_tool_tokens / total_tokens, 4),
                active_path_purity=1.0,
                evidence_density=round(evidence_tokens / total_tokens, 4),
                cold_gap_score=round(cold_by_turn.get(packet.turn_id, 0.0), 4),
                positioned_evidence_score=round(
                    1.0
                    - max(
                        (
                            float(item.axes.get("position_risk") or 0.0)
                            for item in records
                        ),
                        default=0.0,
                    ),
                    4,
                ),
                growth_slope=round(growth_slope, 4),
                taint_exposure=round(tainted_tokens / total_tokens, 4),
                top_reasons=[reason for reason, _count in reason_counts.most_common(5)],
            )
        )
    return result


def session_report(
    session_id: str,
    packets: list[ContextPacketRecord],
    turn_heat: list[TurnHeatRecord],
    findings: list[Finding],
    warnings: list[dict],
) -> dict:
    """Собирает итог по всей сессии для `session_report.json`."""

    return {
        "session_id": session_id,
        "status": "analyzed",
        "turns": len({packet.turn_id for packet in packets}),
        "model_calls": len(packets),
        "max_red_token_share": max(
            (item.red_token_share for item in turn_heat),
            default=0.0,
        ),
        "mean_red_token_share": round(
            sum(item.red_token_share for item in turn_heat) / max(len(turn_heat), 1),
            4,
        ),
        "max_cold_gap_score": max(
            (item.cold_gap_score for item in turn_heat),
            default=0.0,
        ),
        "findings": len(findings),
        "warnings": len(warnings),
        "main_findings": [
            {
                "severity": finding.severity,
                "kind": finding.kind,
                "turn_id": finding.turn_id,
                "title": finding.title,
                "recommendation": finding.recommendation,
                "confidence": finding.confidence,
            }
            for finding in findings[:10]
        ],
        "outputs": {
            "events": "events.jsonl",
            "fragments": "fragments.jsonl",
            "packets": "packets.jsonl",
            "fragment_heat": "fragment_heat.jsonl",
            "fragment_heat_csv": "fragment_heat.csv",
            "turn_heat": "turn_heat.jsonl",
            "turn_heat_csv": "turn_heat.csv",
            "findings": "findings.jsonl",
            "warnings": "warnings.jsonl",
            "session_report": "session_report.json",
            "report": "report.md",
            "heatmap": "heatmap.html",
        },
    }
