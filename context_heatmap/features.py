"""Оси нагрева и холодные дыры для MVP тепловой карты."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .schema import ContextFragmentRecord, ContextPacketRecord, Finding, SessionEvent

PROTECTED_RED_THRESHOLD = 0.75


def clamp(value: float) -> float:
    """Обрезаем score в диапазон 0..1."""

    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class FragmentScore:
    """Расширенная оценка фрагмента с совместимым tuple-интерфейсом."""

    axes: dict[str, float]
    heat: float
    reasons: list[str]
    confidence: float
    ordinary_cost: float
    protected_status: float
    excluded_from_red_token_share: bool
    protected_reasons: list[str]
    context_layer: str
    authority_level: str
    color: str

    def __iter__(self):
        """Сохраняем старое распаковывание: axes, heat, reasons, confidence."""

        yield self.axes
        yield self.heat
        yield self.reasons
        yield self.confidence


def score_fragment(
    fragment: ContextFragmentRecord,
    packet: ContextPacketRecord,
    active_hash_counts: Counter[str],
    recent_counts: Counter[str],
    growth_slope: float,
) -> FragmentScore:
    """Считает оси и итоговый heat для одного фрагмента."""

    axes, ordinary_heat, ordinary_reasons, confidence = _ordinary_score_fragment(
        fragment,
        packet,
        active_hash_counts,
        recent_counts,
        growth_slope,
    )
    context_layer = fragment.context_layer or "unknown"
    authority_level = fragment.authority_level or "unknown"
    if context_layer == "normative":
        protected_axes = _normative_scores(fragment, active_hash_counts)
        protected_status = max(protected_axes.values(), default=0.0)
        protected_reasons = _protected_reasons(protected_axes)
        heat = protected_status
        excluded = protected_status < PROTECTED_RED_THRESHOLD
        reasons = list(protected_reasons)
        if not reasons and ordinary_heat >= 0.25:
            reasons.append("protected_context_cost")
        axes.update(protected_axes)
        return FragmentScore(
            axes=axes,
            heat=heat,
            reasons=reasons,
            confidence=confidence,
            ordinary_cost=ordinary_heat,
            protected_status=protected_status,
            excluded_from_red_token_share=excluded,
            protected_reasons=protected_reasons,
            context_layer=context_layer,
            authority_level=authority_level,
            color=_protected_color(context_layer, ordinary_heat, protected_status),
        )
    if context_layer == "goal" and fragment.goal_role != "attached_data":
        protected_axes = _goal_scores(fragment)
        protected_status = max(
            1.0 - protected_axes["goal_integrity_score"],
            protected_axes["goal_supersession_score"],
            protected_axes["goal_conflict_score"],
            protected_axes["goal_overhang_score"],
            protected_axes["goal_cold_gap_score"],
            protected_axes["attached_data_taint_score"],
        )
        protected_reasons = _protected_reasons(protected_axes)
        heat = protected_status
        excluded = protected_status < PROTECTED_RED_THRESHOLD
        reasons = list(protected_reasons)
        if not reasons and ordinary_heat >= 0.25:
            reasons.append("protected_goal_anchor_cost")
        axes.update(protected_axes)
        return FragmentScore(
            axes=axes,
            heat=heat,
            reasons=reasons,
            confidence=confidence,
            ordinary_cost=ordinary_heat,
            protected_status=protected_status,
            excluded_from_red_token_share=excluded,
            protected_reasons=protected_reasons,
            context_layer=context_layer,
            authority_level=authority_level,
            color=_protected_color(context_layer, ordinary_heat, protected_status),
        )
    if fragment.goal_role == "attached_data":
        attached_taint = _metadata_score(fragment, "attached_data_taint_score")
        axes["attached_data_taint_score"] = attached_taint
        if attached_taint >= 0.50 and "attached_data_taint" not in ordinary_reasons:
            ordinary_reasons.append("attached_data_taint")
        ordinary_heat = clamp(max(ordinary_heat, attached_taint))
    return FragmentScore(
        axes=axes,
        heat=ordinary_heat,
        reasons=ordinary_reasons,
        confidence=confidence,
        ordinary_cost=ordinary_heat,
        protected_status=0.0,
        excluded_from_red_token_share=False,
        protected_reasons=[],
        context_layer=context_layer,
        authority_level=authority_level,
        color=_heat_color_name(ordinary_heat),
    )


def _ordinary_score_fragment(
    fragment: ContextFragmentRecord,
    packet: ContextPacketRecord,
    active_hash_counts: Counter[str],
    recent_counts: Counter[str],
    growth_slope: float,
) -> tuple[dict[str, float], float, list[str], float]:
    """Сохраняем исходную формулу heat для рабочего контекста."""

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


def _normative_scores(
    fragment: ContextFragmentRecord,
    active_hash_counts: Counter[str],
) -> dict[str, float]:
    """Считаем protected status для правил, а не цену их длины."""

    conflict = max(
        _metadata_score(fragment, "instruction_conflict_score"),
        0.90 if fragment.applicability == "conflicting" else 0.0,
    )
    staleness = max(
        _metadata_score(fragment, "instruction_staleness_score"),
        0.85
        if fragment.validity == "stale"
        or fragment.stability == "superseded"
        or fragment.applicability == "superseded"
        else 0.0,
    )
    duplication = max(
        _metadata_score(fragment, "instruction_duplication_score"),
        0.85
        if fragment.content_hash and active_hash_counts[fragment.content_hash] > 1
        else 0.0,
    )
    scope = max(
        _metadata_score(fragment, "instruction_scope_score"),
        1.0 if fragment.applicability == "wrong_project" else 0.0,
        0.65 if fragment.applicability == "inactive_role" else 0.0,
    )
    integrity = max(
        _metadata_score(fragment, "instruction_integrity_score"),
        1.0 if fragment.authority_level in {"external", "user", "assistant"} else 0.0,
        0.30 if not fragment.content_hash else 0.0,
    )
    taint = max(
        _metadata_score(fragment, "instruction_taint_score"),
        _taint(fragment),
        1.0 if fragment.authority_level == "external" else 0.0,
    )
    return {
        "instruction_conflict_score": clamp(conflict),
        "instruction_staleness_score": clamp(staleness),
        "instruction_duplication_score": clamp(duplication),
        "instruction_scope_score": clamp(scope),
        "instruction_integrity_score": clamp(integrity),
        "instruction_taint_score": clamp(taint),
    }


def _goal_scores(fragment: ContextFragmentRecord) -> dict[str, float]:
    """Считаем protected status для активной цели пользователя."""

    integrity = _metadata_score(fragment, "goal_integrity_score", default=1.0)
    if fragment.metadata.get("lost_acceptance_criteria"):
        integrity = min(integrity, 0.20)
    supersession = max(
        _metadata_score(fragment, "goal_supersession_score"),
        0.90
        if fragment.stability == "superseded"
        or fragment.applicability == "superseded"
        else 0.0,
    )
    conflict = max(
        _metadata_score(fragment, "goal_conflict_score"),
        0.85 if fragment.applicability == "conflicting" else 0.0,
    )
    overhang = max(
        _metadata_score(fragment, "goal_overhang_score"),
        0.75
        if fragment.applicability in {"completed", "inactive_role", "superseded"}
        else 0.0,
    )
    return {
        "goal_integrity_score": clamp(integrity),
        "goal_supersession_score": clamp(supersession),
        "goal_conflict_score": clamp(conflict),
        "goal_overhang_score": clamp(overhang),
        "goal_cold_gap_score": _metadata_score(fragment, "goal_cold_gap_score"),
        "attached_data_taint_score": _metadata_score(
            fragment,
            "attached_data_taint_score",
        ),
    }


def _metadata_score(
    fragment: ContextFragmentRecord,
    key: str,
    *,
    default: float = 0.0,
) -> float:
    """Достаём числовой score из metadata без доверия внешнему типу."""

    value = fragment.metadata.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return clamp(float(value))
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return default


def _protected_reasons(scores: dict[str, float]) -> list[str]:
    """Переводим protected metrics в стабильные причины отчета."""

    reasons = []
    reason_by_score = {
        "instruction_conflict_score": "instruction_conflict",
        "instruction_staleness_score": "instruction_stale_or_superseded",
        "instruction_duplication_score": "instruction_duplicate",
        "instruction_scope_score": "instruction_scope_mismatch",
        "instruction_integrity_score": "instruction_integrity_problem",
        "instruction_taint_score": "instruction_tainted_or_untrusted",
        "goal_integrity_score": "goal_integrity_loss",
        "goal_supersession_score": "goal_superseded",
        "goal_conflict_score": "goal_conflict",
        "goal_overhang_score": "goal_overhang",
        "goal_cold_gap_score": "goal_cold_gap",
        "attached_data_taint_score": "attached_data_taint",
    }
    for key, reason in reason_by_score.items():
        value = scores.get(key)
        if value is None:
            continue
        if key == "goal_integrity_score":
            if 1.0 - value >= 0.50:
                reasons.append(reason)
            continue
        if value >= 0.50:
            reasons.append(reason)
    return reasons


def _protected_color(
    context_layer: str,
    ordinary_heat: float,
    protected_status: float,
) -> str:
    """Цвет protected-фрагмента зависит от проблемы, а не только от цены."""

    if protected_status >= 0.75:
        return "red"
    if protected_status >= 0.50:
        return "orange"
    if ordinary_heat >= 0.25:
        return "yellow"
    if context_layer == "goal":
        return "green"
    return "gray"


def _heat_color_name(value: float) -> str:
    """Называем старые пороги heat теми же цветами, что и HTML."""

    if value < 0.25:
        return "green"
    if value < 0.50:
        return "yellow"
    if value < 0.75:
        return "orange"
    return "red"


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
