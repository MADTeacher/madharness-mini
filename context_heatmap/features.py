"""Оси нагрева и холодные дыры для MVP тепловой карты."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .schema import ContextFragmentRecord, ContextPacketRecord, Finding, SessionEvent


def clamp(value: float) -> float:
    """Обрезаем score в диапазон 0..1."""

    return max(0.0, min(1.0, value))


def score_fragment(
    fragment: ContextFragmentRecord,
    packet: ContextPacketRecord,
    active_hash_counts: Counter[str],
    recent_counts: Counter[str],
    growth_slope: float,
) -> tuple[dict[str, float], float, list[str], float]:
    """Считает оси и итоговый heat для одного фрагмента."""

    token_share = fragment.tokens / max(packet.input_tokens, 1)
    window_fill = (
        packet.input_tokens / packet.context_window_tokens
        if packet.context_window_tokens
        else 0.0
    )
    repeat_factor = min(recent_counts[fragment.fragment_id] / 5, 1.0)
    pressure = clamp(
        0.45 * math.sqrt(max(token_share, 0.0) / 0.05)
        + 0.25 * window_fill
        + 0.20 * repeat_factor
        + 0.10 * growth_slope
    )
    low_utility = _low_utility(fragment)
    staleness = _staleness(fragment)
    duplication = _duplication(fragment, active_hash_counts, recent_counts)
    position_risk = _position_risk(fragment, packet, window_fill)
    taint = _taint(fragment)
    branch_mix = 0.0
    compression_risk = _compression_risk(fragment, packet)
    axes = {
        "pressure": pressure,
        "low_utility": low_utility,
        "staleness": staleness,
        "duplication": duplication,
        "position_risk": position_risk,
        "taint": taint,
        "branch_mix": branch_mix,
        "compression_risk": compression_risk,
    }
    risk = max(
        low_utility * 0.70,
        staleness,
        duplication * 0.80,
        position_risk,
        taint,
        branch_mix,
        compression_risk,
    )
    impact = _impact(fragment)
    confidence = min(packet.reconstruction_confidence, _fragment_confidence(fragment))
    heat = clamp(pressure * (0.35 + 0.65 * risk) * impact * confidence)
    return axes, heat, _reasons(axes, fragment), confidence


def detect_cold_gaps(events: list[SessionEvent]) -> list[Finding]:
    """Ищет правки файлов без актуального чтения перед действием."""

    reads_by_path: dict[str, list[int]] = defaultdict(list)
    writes_by_path: dict[str, list[int]] = defaultdict(list)
    known_existing_paths: set[str] = set()
    findings: list[Finding] = []
    counter = 1
    for event_order, event in enumerate(events):
        if event.event_type != "tool_result":
            continue
        tool = str(event.payload.get("tool") or "")
        args = event.payload.get("args")
        if not isinstance(args, dict):
            continue
        observation = event.payload.get("observation")
        if tool == "list_files" and isinstance(observation, dict):
            known_existing_paths.update(
                str(path)
                for path in observation.get("files") or []
                if isinstance(path, str)
            )
        if tool == "read_file" and isinstance(args.get("path"), str):
            reads_by_path[args["path"]].append(event_order)
            known_existing_paths.add(args["path"])
        for path, requires_read in _write_operations(tool, args, known_existing_paths):
            previous_write = max(writes_by_path[path], default=-1)
            has_current_read = any(
                previous_write < read_order < event_order
                for read_order in reads_by_path.get(path, [])
            )
            writes_by_path[path].append(event_order)
            known_existing_paths.add(path)
            if has_current_read or not requires_read:
                continue
            findings.append(
                Finding(
                    finding_id=f"find-{counter:03d}",
                    session_id=event.session_id,
                    turn_id=event.turn_id,
                    severity="medium",
                    kind="cold_gap",
                    title="Правка файла без актуального чтения",
                    explanation=(
                        f"Перед действием `{tool}` для `{path}` в trace нет "
                        "актуального `read_file` после предыдущей правки."
                    ),
                    fragment_ids=[],
                    event_ids=[event.event_id],
                    recommendation=(
                        "Перед следующей правкой перечитать файл или добавить "
                        "в trace доказательство, что текущее состояние известно."
                    ),
                    confidence=0.72,
                    scores={"cold_gap_score": 0.72},
                )
            )
            counter += 1
    return findings


def _write_operations(
    tool: str,
    args: dict[str, Any],
    known_existing_paths: set[str],
) -> list[tuple[str, bool]]:
    """Возвращаем path и признак, нужен ли актуальный `read_file`.

    Первичное создание файла не является cold gap: читать ещё нечего. А вот
    повторная запись, известный существующий файл, Update/Delete patch и source
    при Move требуют доказательства свежего состояния.
    """

    if tool == "write_file" and isinstance(args.get("path"), str):
        path = args["path"]
        return [(path, path in known_existing_paths)]
    if tool != "apply_patch" or not isinstance(args.get("patch"), str):
        return []
    operations: list[tuple[str, bool]] = []
    pattern = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")
    move_pattern = re.compile(r"^\*\*\* Move to: (.+)$")
    for line in args["patch"].splitlines():
        match = pattern.match(line)
        if match:
            operation, path = match.groups()
            operations.append((path.strip(), operation in {"Update", "Delete"}))
            continue
        move_match = move_pattern.match(line)
        if move_match:
            operations.append((move_match.group(1).strip(), False))
    return sorted(set(operations))


def _low_utility(fragment: ContextFragmentRecord) -> float:
    weights = {
        "system_instruction": 0.05,
        "developer_instruction": 0.08,
        "user_message": 0.05,
        "file_snippet": 0.18,
        "test_result": 0.16,
        "tool_output": 0.38,
        "assistant_message": 0.42,
        "tool_schema": 0.35,
        "context_fragment": 0.22,
    }
    return weights.get(fragment.source_type, 0.50)


def _staleness(fragment: ContextFragmentRecord) -> float:
    if fragment.validity == "stale":
        return 0.85
    if fragment.source_type in {"test_result", "file_snippet"} and fragment.metadata.get("legacy"):
        return 0.25
    return 0.0


def _duplication(
    fragment: ContextFragmentRecord,
    active_hash_counts: Counter[str],
    recent_counts: Counter[str],
) -> float:
    score = 0.0
    if fragment.content_hash and active_hash_counts[fragment.content_hash] > 1:
        score = max(score, 0.85)
    if recent_counts[fragment.fragment_id] > 1:
        score = max(score, min(recent_counts[fragment.fragment_id] / 5, 1.0) * 0.65)
    return score


def _position_risk(
    fragment: ContextFragmentRecord,
    packet: ContextPacketRecord,
    window_fill: float,
) -> float:
    match = next(
        (item for item in packet.fragments if item.fragment_id == fragment.fragment_id),
        None,
    )
    if not match or not packet.input_tokens:
        return 0.0
    midpoint = (match.position_start + match.position_end) / 2 / packet.input_tokens
    middle_risk = 1 - abs(2 * midpoint - 1)
    criticality = _criticality(fragment)
    return clamp(middle_risk * criticality * window_fill)


def _taint(fragment: ContextFragmentRecord) -> float:
    if fragment.taint == "secret":
        return 1.0
    if fragment.taint in {"external_text", "possible_injection"}:
        return 0.8
    if fragment.taint == "unknown" and fragment.source_type == "tool_output":
        return 0.18
    if fragment.source_type == "tool_output":
        return 0.10
    return 0.0


def _compression_risk(
    fragment: ContextFragmentRecord,
    packet: ContextPacketRecord,
) -> float:
    if fragment.source_type == "compaction_summary" and not fragment.metadata.get("source_ref"):
        return 0.75
    if "legacy_trace_without_context_packet" in packet.warnings:
        return 0.20
    return 0.0


def _impact(fragment: ContextFragmentRecord) -> float:
    if fragment.source_type in {"system_instruction", "developer_instruction", "user_message"}:
        return 1.0
    if fragment.source_type in {"file_snippet", "test_result", "tool_output"}:
        return 0.9
    if fragment.source_type == "tool_schema":
        return 0.75
    return 0.7


def _criticality(fragment: ContextFragmentRecord) -> float:
    if fragment.source_type in {"user_message", "system_instruction", "developer_instruction"}:
        return 0.9
    if fragment.source_type in {"test_result", "file_snippet"}:
        return 0.8
    if fragment.source_type == "tool_output":
        return 0.55
    return 0.35


def _fragment_confidence(fragment: ContextFragmentRecord) -> float:
    value = fragment.metadata.get("confidence")
    if isinstance(value, int | float):
        return float(value)
    if fragment.metadata.get("legacy"):
        return 0.55
    return 0.85


def _reasons(axes: dict[str, float], fragment: ContextFragmentRecord) -> list[str]:
    reasons: list[str] = []
    thresholds = {
        "pressure": "large_or_repeated_fragment",
        "low_utility": "weak_action_link",
        "staleness": "possibly_stale",
        "duplication": "duplicate_context",
        "position_risk": "middle_position",
        "taint": "tainted_or_untrusted",
        "compression_risk": "compression_or_reconstruction_risk",
    }
    for axis, reason in thresholds.items():
        if axes.get(axis, 0.0) >= 0.50:
            reasons.append(reason)
    if fragment.source_type == "tool_schema" and axes["pressure"] >= 0.25:
        reasons.append("tool_schema_budget")
    return reasons
