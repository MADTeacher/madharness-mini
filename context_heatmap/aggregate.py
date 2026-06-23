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
    pressure_by_turn: dict[int, float] = {}
    for heat in fragment_heat:
        heat_by_call.setdefault(heat.model_call_id, []).append(heat)
    for finding in findings:
        if finding.kind == "cold_gap":
            cold_by_turn[finding.turn_id] = max(
                cold_by_turn.get(finding.turn_id, 0.0),
                float(finding.scores.get("cold_gap_score") or finding.confidence),
            )
        elif finding.kind == "window_pressure":
            pressure_by_turn[finding.turn_id] = max(
                pressure_by_turn.get(finding.turn_id, 0.0),
                float(
                    finding.scores.get("window_pressure_score") or finding.confidence
                ),
            )

    result: list[TurnHeatRecord] = []
    previous_tokens = 0
    for packet in packets:
        records = heat_by_call.get(packet.model_call_id, [])
        token_by_fragment = {item.fragment_id: item.tokens for item in packet.fragments}
        total_tokens = max(packet.input_tokens, sum(token_by_fragment.values()), 1)
        red_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if item.heat >= 0.75 and not item.excluded_from_red_token_share
        )
        stale_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if float(item.axes.get("staleness") or 0.0) >= 0.50
            or float(item.axes.get("instruction_staleness_score") or 0.0) >= 0.50
        )
        raw_tool_tokens = sum(
            fragment.tokens
            for fragment in packet.fragments
            if fragment.source_type == "tool_output"
        )
        # Накопленная история ответов ассистента: отдельный сигнал window_pressure,
        # не покрывается red_token_share (отдельные фрагменты намеренно низкого heat).
        assistant_tokens = sum(
            fragment.tokens
            for fragment in packet.fragments
            if fragment.source_type == "assistant_message"
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
            or float(item.axes.get("instruction_taint_score") or 0.0) >= 0.50
            or float(item.axes.get("attached_data_taint_score") or 0.0) >= 0.50
        )
        fixed_instruction_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if item.context_layer == "normative"
        )
        goal_anchor_tokens = sum(
            token_by_fragment.get(item.fragment_id, 0)
            for item in records
            if item.context_layer == "goal"
        )
        instruction_conflict_score = _max_axis(records, "instruction_conflict_score")
        instruction_staleness_score = _max_axis(records, "instruction_staleness_score")
        instruction_duplication_score = _max_axis(records, "instruction_duplication_score")
        instruction_scope_score = _max_axis(records, "instruction_scope_score")
        instruction_integrity_score = _max_axis(records, "instruction_integrity_score")
        instruction_taint_score = _max_axis(records, "instruction_taint_score")
        goal_integrity_score = _min_axis(records, "goal_integrity_score", default=1.0)
        goal_supersession_score = _max_axis(records, "goal_supersession_score")
        goal_conflict_score = _max_axis(records, "goal_conflict_score")
        goal_overhang_score = _max_axis(records, "goal_overhang_score")
        goal_cold_gap_score = _max_axis(records, "goal_cold_gap_score")
        attached_data_taint_score = _max_axis(records, "attached_data_taint_score")
        normative_status = max(
            instruction_conflict_score,
            instruction_staleness_score,
            instruction_duplication_score,
            instruction_scope_score,
            instruction_integrity_score,
            instruction_taint_score,
        )
        goal_status = max(
            1.0 - goal_integrity_score,
            goal_supersession_score,
            goal_conflict_score,
            goal_overhang_score,
            goal_cold_gap_score,
            attached_data_taint_score,
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
                assistant_share=round(assistant_tokens / total_tokens, 4),
                active_path_purity=1.0,
                evidence_density=round(evidence_tokens / total_tokens, 4),
                cold_gap_score=round(cold_by_turn.get(packet.turn_id, 0.0), 4),
                window_pressure_score=round(
                    pressure_by_turn.get(packet.turn_id, 0.0), 4
                ),
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
                fixed_instruction_cost=round(
                    fixed_instruction_tokens / total_tokens,
                    4,
                ),
                goal_anchor_cost=round(goal_anchor_tokens / total_tokens, 4),
                normative_status=round(normative_status, 4),
                goal_status=round(goal_status, 4),
                instruction_conflict_score=round(instruction_conflict_score, 4),
                instruction_staleness_score=round(instruction_staleness_score, 4),
                instruction_duplication_score=round(instruction_duplication_score, 4),
                instruction_scope_score=round(instruction_scope_score, 4),
                instruction_integrity_score=round(instruction_integrity_score, 4),
                instruction_taint_score=round(instruction_taint_score, 4),
                goal_integrity_score=round(goal_integrity_score, 4),
                goal_supersession_score=round(goal_supersession_score, 4),
                goal_conflict_score=round(goal_conflict_score, 4),
                goal_overhang_score=round(goal_overhang_score, 4),
                goal_cold_gap_score=round(goal_cold_gap_score, 4),
                attached_data_taint_score=round(attached_data_taint_score, 4),
                top_reasons=[reason for reason, _count in reason_counts.most_common(5)],
            )
        )
    return result


def _max_axis(records: list[FragmentHeatRecord], axis: str) -> float:
    """Берём худший score по обращению к модели."""

    return max((float(item.axes.get(axis) or 0.0) for item in records), default=0.0)


def _min_axis(
    records: list[FragmentHeatRecord],
    axis: str,
    *,
    default: float,
) -> float:
    """Для integrity важен минимальный сохранённый показатель."""

    values = [float(item.axes.get(axis)) for item in records if axis in item.axes]
    return min(values, default=default)


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
        "max_window_pressure_score": max(
            (item.window_pressure_score for item in turn_heat),
            default=0.0,
        ),
        "max_assistant_share": max(
            (item.assistant_share for item in turn_heat),
            default=0.0,
        ),
        "mean_assistant_share": round(
            sum(item.assistant_share for item in turn_heat) / max(len(turn_heat), 1),
            4,
        ),
        "max_fixed_instruction_cost": max(
            (item.fixed_instruction_cost for item in turn_heat),
            default=0.0,
        ),
        "max_goal_anchor_cost": max(
            (item.goal_anchor_cost for item in turn_heat),
            default=0.0,
        ),
        "max_normative_status": max(
            (item.normative_status for item in turn_heat),
            default=0.0,
        ),
        "max_goal_status": max(
            (item.goal_status for item in turn_heat),
            default=0.0,
        ),
        "max_instruction_scope_score": max(
            (item.instruction_scope_score for item in turn_heat),
            default=0.0,
        ),
        "max_goal_supersession_score": max(
            (item.goal_supersession_score for item in turn_heat),
            default=0.0,
        ),
        "max_attached_data_taint_score": max(
            (item.attached_data_taint_score for item in turn_heat),
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
            "heatmap_png": "heatmap.png",
            "context_window_png": "context_window.png",
        },
    }
