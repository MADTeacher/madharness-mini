"""Запись машинных и Markdown-отчетов тепловой карты."""

from __future__ import annotations

from pathlib import Path

from .io import write_csv, write_json, write_jsonl
from .png import render_png_summary
from .schema import AnalysisResult


def write_analysis_outputs(result: AnalysisResult, out_dir: Path) -> None:
    """Пишет все артефакты анализа одной сессии."""

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "events.jsonl", [event.to_dict() for event in result.events])
    write_jsonl(
        out_dir / "fragments.jsonl",
        [fragment.to_dict() for fragment in result.fragments],
    )
    write_jsonl(
        out_dir / "packets.jsonl",
        [packet.to_dict() for packet in result.packets],
    )
    heat_rows = [heat.to_dict() for heat in result.fragment_heat]
    turn_rows = [turn.to_dict() for turn in result.turn_heat]
    write_jsonl(out_dir / "fragment_heat.jsonl", heat_rows)
    write_jsonl(out_dir / "turn_heat.jsonl", turn_rows)
    write_jsonl(
        out_dir / "findings.jsonl",
        [finding.to_dict() for finding in result.findings],
    )
    write_jsonl(out_dir / "warnings.jsonl", result.warnings)
    write_json(out_dir / "session_report.json", result.session_report)
    render_png_summary(
        result.session_report,
        [packet.to_dict() for packet in result.packets],
        turn_rows,
        [finding.to_dict() for finding in result.findings],
        out_dir / "heatmap.png",
    )
    write_csv(
        out_dir / "fragment_heat.csv",
        heat_rows,
        [
            "session_id",
            "model_call_id",
            "fragment_id",
            "heat",
            "confidence",
            "context_layer",
            "authority_level",
            "ordinary_cost",
            "protected_status",
            "excluded_from_red_token_share",
            "color",
            "axes",
            "reasons",
            "protected_reasons",
        ],
    )
    write_csv(
        out_dir / "turn_heat.csv",
        turn_rows,
        [
            "session_id",
            "model_call_id",
            "turn_id",
            "red_token_share",
            "stale_token_share",
            "raw_tool_share",
            "assistant_share",
            "evidence_density",
            "cold_gap_score",
            "window_pressure_score",
            "taint_exposure",
            "fixed_instruction_cost",
            "goal_anchor_cost",
            "normative_status",
            "goal_status",
            "instruction_scope_score",
            "goal_supersession_score",
            "attached_data_taint_score",
            "top_reasons",
        ],
    )
    (out_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: AnalysisResult) -> str:
    """Создает краткий человекочитаемый отчет."""

    report = result.session_report
    lines = [
        f"# Context Heatmap Report: `{result.session_id}`",
        "",
        "## Краткое состояние",
        "",
        f"- model calls: {report['model_calls']}",
        f"- max red token share: {report['max_red_token_share']}",
        f"- max assistant share: {report.get('max_assistant_share', 0)}",
        f"- max cold gap score: {report['max_cold_gap_score']}",
        f"- max window pressure score: {report.get('max_window_pressure_score', 0)}",
        f"- max fixed instruction cost: {report.get('max_fixed_instruction_cost', 0)}",
        f"- max goal anchor cost: {report.get('max_goal_anchor_cost', 0)}",
        f"- max normative status: {report.get('max_normative_status', 0)}",
        f"- max goal status: {report.get('max_goal_status', 0)}",
        f"- findings: {report['findings']}",
        f"- warnings: {report['warnings']}",
        "",
        "## Самые горячие обращения",
        "",
    ]
    hottest = sorted(
        result.turn_heat,
        key=lambda item: (
            item.red_token_share,
            item.cold_gap_score,
            item.window_pressure_score,
        ),
        reverse=True,
    )[:5]
    if not hottest:
        lines.append("Нет обращений к модели для анализа.")
    for item in hottest:
        lines.append(
            "- turn "
            f"{item.turn_id}: red={item.red_token_share}, "
            f"assist={item.assistant_share}, "
            f"cold={item.cold_gap_score}, "
            f"pressure={item.window_pressure_score}, "
            f"instruction_cost={item.fixed_instruction_cost}, "
            f"goal_cost={item.goal_anchor_cost}, "
            f"reasons={', '.join(item.top_reasons) or 'none'}"
        )
    lines.extend(["", "## Findings", ""])
    if not result.findings:
        lines.append("Критичных находок не найдено.")
    for finding in result.findings[:10]:
        lines.extend(
            [
                f"### {finding.severity}: {finding.title}",
                "",
                finding.explanation,
                "",
                f"Recommendation: {finding.recommendation}",
                "",
            ]
        )
    lines.extend(
        [
            "## Ограничения анализа",
            "",
            "Пассивная карта показывает подозрительные участки trace, но не "
            "доказывает причинность без replay-экспериментов и калибровки на корпусе.",
            "",
        ]
    )
    return "\n".join(lines)
