"""Запись машинных и Markdown-отчетов тепловой карты."""

from __future__ import annotations

from pathlib import Path

from .io import write_csv, write_json, write_jsonl
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
    write_csv(
        out_dir / "fragment_heat.csv",
        heat_rows,
        ["session_id", "model_call_id", "fragment_id", "heat", "confidence", "axes", "reasons"],
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
            "evidence_density",
            "cold_gap_score",
            "taint_exposure",
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
        f"- max cold gap score: {report['max_cold_gap_score']}",
        f"- findings: {report['findings']}",
        f"- warnings: {report['warnings']}",
        "",
        "## Самые горячие обращения",
        "",
    ]
    hottest = sorted(
        result.turn_heat,
        key=lambda item: (item.red_token_share, item.cold_gap_score),
        reverse=True,
    )[:5]
    if not hottest:
        lines.append("Нет обращений к модели для анализа.")
    for item in hottest:
        lines.append(
            "- turn "
            f"{item.turn_id}: red={item.red_token_share}, "
            f"cold={item.cold_gap_score}, reasons={', '.join(item.top_reasons) or 'none'}"
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
