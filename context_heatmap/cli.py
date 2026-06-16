"""CLI для пассивного анализа тепловой карты контекста."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json, write_jsonl
from .loaders.madharness_trace import load_trace, load_trace_path
from .normalize import load_normalized_events, normalize_input
from .render import render_html_report
from .report import write_analysis_outputs
from .scoring import analyze_events


CRITICAL_WARNING_KINDS = {"invalid_jsonl", "non_object_event"}


def main(argv: list[str] | None = None) -> None:
    """Разбираем команды `context-heatmap`."""

    parser = argparse.ArgumentParser(prog="context-heatmap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)

    normalize = sub.add_parser("normalize")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--out", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--events", required=True)
    analyze.add_argument("--out", required=True)

    batch = sub.add_parser("analyze-batch")
    batch.add_argument("--input", required=True)
    batch.add_argument("--out", required=True)

    render = sub.add_parser("render")
    render.add_argument("--report", required=True)
    render.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "validate":
        _validate(Path(args.input))
    elif args.cmd == "normalize":
        events, warnings = normalize_input(Path(args.input), Path(args.out))
        print(f"normalized events: {len(events)}")
        print(f"warnings: {len(warnings)}")
    elif args.cmd == "analyze":
        events_path = Path(args.events)
        events = load_normalized_events(events_path)
        result = analyze_events(events, _load_sibling_warnings(events_path))
        out_dir = Path(args.out)
        write_analysis_outputs(result, out_dir)
        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        print(f"analyzed model calls: {len(result.packets)}")
        print(f"report: {out_dir / 'session_report.json'}")
    elif args.cmd == "analyze-batch":
        _analyze_batch(Path(args.input), Path(args.out))
    elif args.cmd == "render":
        render_html_report(Path(args.report), Path(args.out))
        print(f"rendered: {args.out}")


def _validate(path: Path) -> None:
    """Проверяет, что input читается и содержит обращения к модели."""

    events, warnings = load_trace_path(path)
    model_calls = sum(1 for event in events if event.event_type == "model_call")
    packet_errors = _context_packet_errors(events)
    critical_warnings = [
        warning
        for warning in warnings
        if str(warning.get("kind") or "") in CRITICAL_WARNING_KINDS
    ]
    print(f"events: {len(events)}")
    print(f"model calls: {model_calls}")
    print(f"warnings: {len(warnings)}")
    print(f"context packet errors: {len(packet_errors)}")
    if warnings:
        for warning in warnings[:5]:
            print(f"warning: {warning.get('kind')}: {warning.get('message', '')}")
    for error in packet_errors[:5]:
        print(f"context packet error: {error}")
    if not events or not model_calls or critical_warnings or packet_errors:
        raise SystemExit(1)


def _load_sibling_warnings(events_path: Path) -> list[dict[str, Any]]:
    """Подхватываем warnings, созданные `normalize`, если они лежат рядом."""

    warnings_path = events_path.parent / "warnings.jsonl"
    if not warnings_path.exists():
        return []
    return read_jsonl(warnings_path)


def _context_packet_errors(events: list[Any]) -> list[str]:
    """Проверяем схему и монотонность `context_packet` без legacy-штрафа."""

    errors: list[str] = []
    for event in events:
        if event.event_type != "model_call":
            continue
        report = event.payload.get("context_report")
        if not isinstance(report, dict):
            continue
        packet = report.get("context_packet")
        if not isinstance(packet, dict):
            continue
        event_label = event.event_id
        if packet.get("version") != 1:
            errors.append(f"{event_label}: context_packet.version must be 1")
        units = packet.get("units")
        if not isinstance(units, list):
            errors.append(f"{event_label}: context_packet.units must be a list")
            continue
        previous_end = 0
        for index, unit in enumerate(units):
            unit_label = f"{event_label}:units[{index}]"
            if not isinstance(unit, dict):
                errors.append(f"{unit_label}: unit must be an object")
                continue
            errors.extend(_context_unit_errors(unit_label, unit, previous_end))
            previous_end = _int_value(unit.get("position_end"), previous_end)
    return errors


def _context_unit_errors(
    unit_label: str,
    unit: dict[str, Any],
    previous_end: int,
) -> list[str]:
    """Проверяем обязательные поля одного unit prompt-индекса."""

    errors: list[str] = []
    required = [
        "unit_id",
        "source_type",
        "source_name",
        "source_ref",
        "tokens_estimate",
        "position_start",
        "position_end",
        "included_because",
        "content_hash",
        "confidence",
    ]
    for key in required:
        if key not in unit:
            errors.append(f"{unit_label}: missing {key}")
    start = _int_value(unit.get("position_start"), 0)
    end = _int_value(unit.get("position_end"), 0)
    tokens = _int_value(unit.get("tokens_estimate"), 0)
    confidence = _float_value(unit.get("confidence"), -1.0)
    if start < previous_end:
        errors.append(f"{unit_label}: position_start is not monotonic")
    if end < start:
        errors.append(f"{unit_label}: position_end is before position_start")
    if tokens < 0:
        errors.append(f"{unit_label}: tokens_estimate is negative")
    if end - start != tokens:
        errors.append(f"{unit_label}: positions do not match tokens_estimate")
    if not 0.0 <= confidence <= 1.0:
        errors.append(f"{unit_label}: confidence must be between 0 and 1")
    if not str(unit.get("content_hash") or ""):
        errors.append(f"{unit_label}: content_hash is empty")
    if unit.get("source_type") == "tool_schema" and not str(
        unit.get("source_ref") or ""
    ).startswith("tools["):
        errors.append(f"{unit_label}: tool_schema source_ref must point to tools[]")
    for key in ("position_start", "position_end", "tokens_estimate"):
        if key in unit and not isinstance(unit.get(key), int):
            errors.append(f"{unit_label}: {key} must be an integer")
    if "confidence" in unit and not isinstance(unit.get("confidence"), int | float):
        errors.append(f"{unit_label}: confidence must be numeric")
    return errors


def _int_value(value: Any, default: int) -> int:
    """Безопасно читаем integer-поле из внешнего trace."""

    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    """Безопасно читаем числовое поле из внешнего trace."""

    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _analyze_batch(input_path: Path, out_dir: Path) -> None:
    """Анализирует каталог trace-файлов по одному отчету на файл."""

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    corpus = []
    for trace_path in trace_files:
        events, warnings = load_trace(trace_path)
        result = analyze_events(events, warnings)
        session_dir = out_dir / trace_path.stem
        write_analysis_outputs(result, session_dir)
        render_html_report(
            session_dir / "session_report.json",
            session_dir / "heatmap.html",
        )
        corpus.append(result.session_report)
    write_jsonl(out_dir / "corpus_report.jsonl", corpus)
    write_json(
        out_dir / "corpus_report.json",
        {
            "sessions": len(corpus),
            "reports": corpus,
            "max_red_token_share": max(
                (item.get("max_red_token_share", 0.0) for item in corpus),
                default=0.0,
            ),
            "max_cold_gap_score": max(
                (item.get("max_cold_gap_score", 0.0) for item in corpus),
                default=0.0,
            ),
        },
    )
    print(f"analyzed sessions: {len(corpus)}")
