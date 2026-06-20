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


# Пороги фиксированных severity/confidence для расширенных cold-gap сигнатур.
# Значения подобраны так, чтобы ловить доказанные провалы контекстной политики
# и при этом не раздувать max_cold_gap_score ложными срабатываниями.
REPEATED_WRITE_GAP_SCORE = 0.72
SPEC_MISSING_GAP_SCORE = 0.72
POST_SUMMARY_GAP_SCORE = 0.70
APPLY_PATCH_STORM_SCORE = 0.60
READ_STORM_SCORE = 0.55

# Порог срабатывания сигнатуры re-read storm: подозрительно много повторных
# чтений одного и того же пути в скользящем окне ходов — компенсация амнезии.
READ_STORM_WINDOW_TURNS = 6
READ_STORM_THRESHOLD = 3

# Порог apply_patch storm: неудачных правок одного пути в скользящем окне.
# Окно шире, чем у read_storm, т.к. между неудачами модель обычно перечитывает
# файл и тратит дополнительные ходы — цикл растягивается во времени.
APPLY_PATCH_STORM_WINDOW_TURNS = 8
APPLY_PATCH_STORM_THRESHOLD = 2

# Минимальная доля от медианного размера чтения, чтобы признать путь
# «документом-спецификацией» (ТЗ, контракт), а не мелким snippet-ом.
SPEC_TOKENS_FACTOR = 3.0
# При малом числе чтений медиана равна самóму размеру, поэтому относительный
# порог не работает: используем абсолютный минимум в символах. ~1 КБ — это
# всё ещё небольшой файл, но уже не одиночный snippet.
SPEC_MIN_READS_FOR_MEDIAN = 3
SPEC_ABSOLUTE_MIN_CHARS = 1000


@dataclass(frozen=True)
class WindowIndex:
    """Предобработанные данные о prompt-пакетах и summarization по ходам.

    Собирается один раз по model_call-событиям; read-only, без скрытого
    состояния — это позволяет тестировать сигнатуры детерминированно.
    """

    # turn_id -> множество путей, представленных в окне на этом ходе
    # (file_snippet-ы, read_file-наблюдения, test_result-ы).
    paths_by_turn: dict[int, frozenset[str]]
    # event_order -> множество путей, свернутых summarization'ом на этом ходе.
    collapsed_by_order: dict[int, frozenset[str]]


def detect_cold_gaps(events: list[SessionEvent]) -> list[Finding]:
    """Ищем холодные дыры: правки и действия без свежего доказательства в окне.

    Ловим пять классов сигнатур (все с kind="cold_gap", различаются
    title/explanation, чтобы пройти фильтр aggregate без правок рендера):

    1. repeated_write — повторная правка того же файла без read_file между
       правками (классический сценарий, был в MVP).
    2. spec_missing — правка, когда ключевой документ-спецификация отсутствует
       в текущем prompt-пакете (свернут summarization'ом или не перечитан).
    3. post_summary — правка сразу после хода, где summarization свернул
       read_file критического документа.
    4. read_storm — аномальная частота повторных чтений одного пути; это не
       классический cold gap, а компенсаторное поведение при потере доверия к
       контексту.
    5. apply_patch_storm — несколько неудачных apply_patch по тому же пути:
       симптом рассинхронизации памяти модели и актуального файла.
    """

    # Индекс prompt-пакетов и summarization по ходам + выявление документов-
    # спецификаций строится по model_call-событиям. Legacy-трассы без
    # context_report не падают: все обращения через .get() с дефолтами.
    window_index = _build_window_index(events)
    spec_paths = _spec_paths(events)

    reads_by_path: dict[str, list[int]] = defaultdict(list)
    writes_by_path: dict[str, list[int]] = defaultdict(list)
    read_turns_by_path: dict[str, list[int]] = defaultdict(list)
    # Сигнатура №5: turn'ы неудачных apply_patch по пути. Успешная правка
    # сбрасывает счётчик — модель исправилась, цикл прерван.
    failed_patch_turns_by_path: dict[str, list[int]] = defaultdict(list)
    known_existing_paths: set[str] = set()
    raw_findings: list[Finding] = []
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
            read_turns_by_path[args["path"]].append(event.turn_id)
            known_existing_paths.add(args["path"])

            # Сигнатура №4: re-read storm. Считаем по turn_id, чтобы соседние
            # повторные чтения в одном ходе не давали ложного срабатывания.
            storm = _read_storm_finding(
                args["path"],
                read_turns_by_path[args["path"]],
                event,
            )
            if storm is not None:
                storm.finding_id = f"find-{counter:03d}"
                counter += 1
                raw_findings.append(storm)

        # Сигнатура №5: apply_patch storm. Успешная правка сбрасывает счётчик
        # неудач по путям из патча (модель исправилась). Неудача — копит turn.
        if tool == "apply_patch" and isinstance(args.get("patch"), str):
            patch_paths = _paths_from_patch(args["patch"])
            observation_ok = bool(observation.get("ok")) if isinstance(observation, dict) else False
            if observation_ok:
                for patch_path in patch_paths:
                    failed_patch_turns_by_path.pop(patch_path, None)
            else:
                for patch_path in patch_paths:
                    failed_patch_turns_by_path[patch_path].append(event.turn_id)
                    patch_storm = _apply_patch_storm_finding(
                        patch_path,
                        failed_patch_turns_by_path[patch_path],
                        event,
                    )
                    if patch_storm is not None:
                        patch_storm.finding_id = f"find-{counter:03d}"
                        counter += 1
                        raw_findings.append(patch_storm)

        for path, requires_read in _write_operations(tool, args, known_existing_paths):
            previous_write = max(writes_by_path[path], default=-1)
            has_current_read = any(
                previous_write < read_order < event_order
                for read_order in reads_by_path.get(path, [])
            )
            writes_by_path[path].append(event_order)
            known_existing_paths.add(path)

            # На одну (event_id, path) берём одну находку с наибольшим score.
            # Сигнатуры spec_missing и post_summary могут срабатывать вместе с
            # repeated_write на одной правке — дедуп прямо при сборе кандидатов.
            candidate = _best_write_finding(
                tool=tool,
                path=path,
                event=event,
                event_order=event_order,
                requires_read=requires_read,
                has_current_read=has_current_read,
                spec_paths=spec_paths,
                window_index=window_index,
                reads_by_path=reads_by_path,
            )
            if candidate is not None:
                candidate.finding_id = f"find-{counter:03d}"
                counter += 1
                raw_findings.append(candidate)

    return raw_findings


def _best_write_finding(
    *,
    tool: str,
    path: str,
    event: SessionEvent,
    event_order: int,
    requires_read: bool,
    has_current_read: bool,
    spec_paths: set[str],
    window_index: WindowIndex,
    reads_by_path: dict[str, list[int]],
) -> Finding | None:
    """Выбираем одну холодную дыру для правки с максимальным confidence.

    Кандидаты опрашиваются от сильного сигнала к слабому; первый непустой
    выигрывает, т.к. пороги confidence упорядочены (repeated_write/spec_missing
    = 0.72 > post_summary = 0.70). Это и есть дедуп: на одну (event_id, path)
    ровно одна находка, без парсинга explanation.
    """

    if requires_read and not has_current_read:
        return _repeated_write_finding(tool, path, event)
    spec_candidate = _spec_missing_finding(
        tool, path, event, spec_paths, window_index
    )
    if spec_candidate is not None:
        return spec_candidate
    return _post_summary_finding(
        event_order, tool, path, event, window_index, reads_by_path
    )


def _repeated_write_finding(
    tool: str, path: str, event: SessionEvent
) -> Finding:
    """Сигнатура №1 (классическая): повторная правка без read между правками."""

    return Finding(
        finding_id="",  # проставляется централизованно в detect_cold_gaps
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
        confidence=REPEATED_WRITE_GAP_SCORE,
        scores={"cold_gap_score": REPEATED_WRITE_GAP_SCORE},
    )


def _spec_missing_finding(
    tool: str,
    path: str,
    event: SessionEvent,
    spec_paths: set[str],
    window_index: WindowIndex,
) -> Finding | None:
    """Сигнатура №2: правка при отсутствии спецификации в текущем окне.

    Если хотя бы один путь из spec_paths не представлен в prompt-пакете того же
    хода (ни полным фрагментом, ни свежим read_file-наблюдением), модель правит
    файл без знания ТЗ. Ловит потерю source-of-truth после summarization.
    """

    if not spec_paths:
        return None
    in_window = window_index.paths_by_turn.get(event.turn_id, frozenset())
    missing = sorted(spec for spec in spec_paths if spec not in in_window)
    if not missing:
        return None
    specs_text = ", ".join(f"`{spec}`" for spec in missing)
    return Finding(
        finding_id="",
        session_id=event.session_id,
        turn_id=event.turn_id,
        severity="medium",
        kind="cold_gap",
        title="Правка при отсутствии спецификации в окне",
        explanation=(
            f"Перед действием `{tool}` для `{path}` в окне нет источника "
            f"спецификации задачи ({specs_text}) — он свёрнут summarization'ом "
            "или никогда не перечитан."
        ),
        fragment_ids=[],
        event_ids=[event.event_id],
        recommendation=(
            "Перечитать ключевой документ задачи перед правкой или закрепить "
            "его как постоянный фрагмент контекста."
        ),
        confidence=SPEC_MISSING_GAP_SCORE,
        scores={"cold_gap_score": SPEC_MISSING_GAP_SCORE},
    )


def _post_summary_finding(
    event_order: int,
    tool: str,
    path: str,
    event: SessionEvent,
    window_index: WindowIndex,
    reads_by_path: dict[str, list[int]],
) -> Finding | None:
    """Сигнатура №3: правка сразу после summarization критического чтения.

    Если на предыдущих ходах summarization свернул read_file-наблюдение, а после
    этого модель правит файл без перечитывания свёрнутого пути — это холодная
    дыра от возрастной эвикции.
    """

    if not window_index.collapsed_by_order:
        return None
    # Момент последнего сворачивания по каждому пути: до текущей правки.
    collapsed_paths_before: list[tuple[int, str]] = []
    for collapse_order, paths in window_index.collapsed_by_order.items():
        if collapse_order >= event_order:
            continue
        for collapsed_path in paths:
            collapsed_paths_before.append((collapse_order, collapsed_path))
    if not collapsed_paths_before:
        return None
    collapsed_paths_before.sort()
    # Берём самый свежий свёрнутый путь, не перечитанный после сворачивания.
    for collapse_order, collapsed_path in reversed(collapsed_paths_before):
        fresh_reads = reads_by_path.get(collapsed_path, [])
        if any(read_order > collapse_order for read_order in fresh_reads):
            # Модель перечитала путь после сворачивания — холодной дыры нет.
            continue
        return Finding(
            finding_id="",
            session_id=event.session_id,
            turn_id=event.turn_id,
            severity="medium",
            kind="cold_gap",
            title="Правка после сворачивания чтения summarization'ом",
            explanation=(
                f"Summarization свернул `read_file` для `{collapsed_path}`, "
                f"затем выполнена правка `{tool}` для `{path}` без "
                "перечитывания свёрнутого документа."
            ),
            fragment_ids=[],
            event_ids=[event.event_id],
            recommendation=(
                "После summarization критических чтений перечитывать файл или "
                "форсировать re-injection его содержимого в prompt-пакет."
            ),
            confidence=POST_SUMMARY_GAP_SCORE,
            scores={"cold_gap_score": POST_SUMMARY_GAP_SCORE},
        )
    return None


def _read_storm_finding(
    path: str,
    read_turns: list[int],
    event: SessionEvent,
) -> Finding | None:
    """Сигнатура №4: аномальная частота повторных чтений одного пути.

    Если путь перечитан >= READ_STORM_THRESHOLD раз за последние
    READ_STORM_WINDOW_TURNS ходов — модель компенсирует потерю доверия к
    контексту. Не классический cold gap, а симптом сломанной политики памяти.
    """

    if len(read_turns) < READ_STORM_THRESHOLD:
        return None
    current_turn = event.turn_id
    window_start = current_turn - READ_STORM_WINDOW_TURNS
    recent = [turn for turn in read_turns if window_start <= turn <= current_turn]
    # Срабатываем ровно один раз — на чтении, которое переполнило порог.
    if len(recent) != READ_STORM_THRESHOLD:
        return None
    return Finding(
        finding_id="",
        session_id=event.session_id,
        turn_id=event.turn_id,
        severity="low",
        kind="cold_gap",
        title="Аномальная частота повторных чтений файла",
        explanation=(
            f"Файл `{path}` перечитан {len(recent)} раз за "
            f"{READ_STORM_WINDOW_TURNS} ходов — признак потери доверия к "
            "контексту после summarization."
        ),
        fragment_ids=[],
        event_ids=[event.event_id],
        recommendation=(
            "Закрепить стабильный фрагмент как постоянный или пересмотреть "
            "политику summarization для этого источника."
        ),
        confidence=READ_STORM_SCORE,
        scores={"cold_gap_score": READ_STORM_SCORE},
    )


def _apply_patch_storm_finding(
    path: str,
    failed_turns: list[int],
    event: SessionEvent,
) -> Finding | None:
    """Сигнатура №5: цикл неудачных apply_patch по одному пути.

    Если apply_patch по пути завершился ошибкой >= APPLY_PATCH_STORM_THRESHOLD
    раз за последние APPLY_PATCH_STORM_WINDOW_TURNS ходов — содержимое файла
    рассинхронизировано с памятью модели, и она буксует. Симптом дороже read
    storm по токенам: каждый неудачный патч — отдельное обращение к модели.
    """

    if len(failed_turns) < APPLY_PATCH_STORM_THRESHOLD:
        return None
    current_turn = event.turn_id
    window_start = current_turn - APPLY_PATCH_STORM_WINDOW_TURNS
    recent = [turn for turn in failed_turns if window_start <= turn <= current_turn]
    # Срабатываем ровно один раз — на неудаче, которая переполнила порог.
    if len(recent) != APPLY_PATCH_STORM_THRESHOLD:
        return None
    return Finding(
        finding_id="",
        session_id=event.session_id,
        turn_id=event.turn_id,
        severity="low",
        kind="cold_gap",
        title="Цикл неудачных правок файла",
        explanation=(
            f"apply_patch для `{path}` завершился ошибкой {len(recent)} раз за "
            f"{APPLY_PATCH_STORM_WINDOW_TURNS} ходов — содержимое файла "
            "рассинхронизировано с памятью модели."
        ),
        fragment_ids=[],
        event_ids=[event.event_id],
        recommendation=(
            "Перечитать файл перед правкой или использовать write_file для "
            "полной перезаписи."
        ),
        confidence=APPLY_PATCH_STORM_SCORE,
        scores={"cold_gap_score": APPLY_PATCH_STORM_SCORE},
    )


def _paths_from_patch(patch: str) -> list[str]:
    """Пути файлов из текста patch в формате Codex.

    Локальная копия парсера (аналог paths_from_patch в madharness_mini/utils),
    чтобы context_heatmap не зависел от harness-кода. Берём пути из строк
    '*** Add|Update|Delete File: <path>'.
    """

    paths: list[str] = []
    pattern = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$")
    for line in patch.splitlines():
        match = pattern.match(line)
        if match:
            paths.append(match.group(1).strip())
    return paths


def _build_window_index(events: list[SessionEvent]) -> WindowIndex:
    """Собираем read-only индекс prompt-пакетов и summarization по ходам.

    Для каждого model_call-события достаём пути, представленные в окне
    (file_snippet, read_file-наблюдение, test_result), и пути, свернутые
    summarization'ом на этом ходе (через соответствие index -> path из
    included_entries[].file_refs). Legacy-трассы без context_report дают пустые
    множества — сигнатуры корректно отрабатывают как «нет данных».
    """

    paths_by_turn: dict[int, frozenset[str]] = {}
    collapsed_by_order: dict[int, frozenset[str]] = {}
    for event_order, event in enumerate(events):
        if event.event_type != "model_call":
            continue
        report = event.payload.get("context_report")
        if not isinstance(report, dict):
            continue
        paths_in_packet: set[str] = set()
        # 1) Пути из units: file_snippet-ы и tool_output-наблюдения несут
        # знание о содержимом файла; tool_schema/assistant_message — нет.
        packet = report.get("context_packet")
        if isinstance(packet, dict):
            for unit in packet.get("units") or []:
                if not isinstance(unit, dict):
                    continue
                source_type = unit.get("source_type") or ""
                if source_type not in {"file_snippet", "tool_output", "test_result"}:
                    continue
                path = _path_from_unit(unit)
                if path:
                    paths_in_packet.add(path)
        # 2) Пути из included_entries через file_refs — самый надёжный источник:
        # harness проставляет kind=read/write для каждой записи истории. Но путь
        # свёрнутой summarization'ом записи НЕ считаем присутствующим в окне:
        # контент заменён дайджестом-указателем, реальное содержимое файла ушло.
        history = report.get("history")
        index_to_path: dict[int, str] = {}
        collapsed_indexes: set[int] = set()
        collapsed_now: set[str] = set()
        if isinstance(history, dict):
            for collapsed_entry in history.get("summarized_old_entries") or []:
                if isinstance(collapsed_entry, dict):
                    collapsed_index = collapsed_entry.get("index")
                    if isinstance(collapsed_index, int):
                        collapsed_indexes.add(collapsed_index)
            for entry in history.get("included_entries") or []:
                if not isinstance(entry, dict):
                    continue
                entry_index = entry.get("index")
                is_summarized = (
                    isinstance(entry_index, int) and entry_index in collapsed_indexes
                )
                for ref in entry.get("file_refs") or []:
                    if isinstance(ref, dict) and ref.get("kind") == "read":
                        ref_path = ref.get("path")
                        if isinstance(ref_path, str):
                            if isinstance(entry_index, int):
                                index_to_path[entry_index] = ref_path
                            # Свёрнутое чтение — это указатель, а не контент:
                            # модель знает, что читала, но не видит содержимое.
                            if not is_summarized:
                                paths_in_packet.add(ref_path)
            # 3) Свёрнутые summarization'ом записи: восстанавливаем путь по
            # соответствию index -> path, собранному из included_entries.
            for collapsed_index in collapsed_indexes:
                collapsed_path = index_to_path.get(collapsed_index)
                if collapsed_path:
                    collapsed_now.add(collapsed_path)
        paths_by_turn[event.turn_id] = frozenset(paths_in_packet)
        if collapsed_now:
            collapsed_by_order[event_order] = frozenset(collapsed_now)
    return WindowIndex(
        paths_by_turn=paths_by_turn,
        collapsed_by_order=collapsed_by_order,
    )


def _path_from_unit(unit: dict[str, Any]) -> str | None:
    """Достаём путь файла из prompt-unit: сначала metadata.path, потом source_ref."""

    metadata = unit.get("metadata")
    if isinstance(metadata, dict):
        path = metadata.get("path")
        if isinstance(path, str):
            return path
    source_ref = unit.get("source_ref")
    if isinstance(source_ref, str):
        if source_ref.startswith("history["):
            # history[N].messages[M] — observation; путь уже проверили в metadata.
            return None
        if "/" in source_ref:
            # source_ref часто совпадает с путём файла (read_file кладёт путь).
            return source_ref
    return None


def _spec_paths(events: list[SessionEvent]) -> set[str]:
    """Выявляем пути-спецификации: ключевые документы задачи (ТЗ, контракты).

    Эвристика без ML. Путь считается спецификацией, если он прочитан read_file'ом
    до первой правки и крупнее порога размера. Порог выбираем адаптивно:
      • чтений >= SPEC_MIN_READS_FOR_MEDIAN — берём медианный размер, умноженный
        на SPEC_TOKENS_FACTOR (крупнее среднего чтения);
      • иначе — абсолютный SPEC_ABSOLUTE_MIN_CHARS, т.к. на малом числе чтений
        медиана равна самóму размеру и относительный порог не работает.
    Для flappy-трасс это устойчиво даёт flappy-bird-prompt.md (единственное
    крупное чтение в начале сессии).
    """

    read_sizes: dict[str, int] = {}
    first_write_order: int | None = None
    read_order_by_path: dict[str, int] = {}
    for event_order, event in enumerate(events):
        if event.event_type != "tool_result":
            continue
        tool = str(event.payload.get("tool") or "")
        args = event.payload.get("args")
        if not isinstance(args, dict):
            continue
        if tool == "read_file" and isinstance(args.get("path"), str):
            path = args["path"]
            read_order_by_path.setdefault(path, event_order)
            size = _estimate_observation_size(event.payload.get("observation"))
            if size > read_sizes.get(path, 0):
                read_sizes[path] = size
        elif tool in {"write_file", "apply_patch"} and first_write_order is None:
            first_write_order = event_order
    if not read_sizes:
        return set()
    sizes = sorted(read_sizes.values())
    if len(sizes) >= SPEC_MIN_READS_FOR_MEDIAN:
        median = sizes[len(sizes) // 2]
        threshold = max(median * SPEC_TOKENS_FACTOR, 1)
    else:
        # Мало чтений — относительный порог бессмысленен, берём абсолютный.
        threshold = SPEC_ABSOLUTE_MIN_CHARS
    specs: set[str] = set()
    for path, size in read_sizes.items():
        if size < threshold:
            continue
        read_order = read_order_by_path.get(path)
        # Спецификация должна быть прочитана до первой правки — это документ,
        # задающий задачу, а не реактивное чтение уже изменённого файла.
        if first_write_order is not None and read_order is not None and read_order > first_write_order:
            continue
        specs.add(path)
    return specs


def _estimate_observation_size(observation: Any) -> int:
    """Грубая оценка размера наблюдения read_file в символах."""

    if isinstance(observation, dict):
        content = observation.get("content")
        if isinstance(content, str):
            return len(content)
        # payload уже мог быть свёрнут: тогда size неизвестен, возвращаем 0.
    return 0


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
    # Intra-packet дубликат (один hash встречается >1 раза в одном окне) —
    # значимый сигнал для всех типов фрагментов.
    if fragment.content_hash and active_hash_counts[fragment.content_hash] > 1:
        score = max(score, 0.85)
    # Inter-turn повтор того же fragment_id — для tool_schema это ожидаемо:
    # схемы инструментов статичны и подаются каждый ход. Считаем их дубликатом
    # только когда одинаковая схема встретилась дважды внутри одного пакета.
    if fragment.source_type != "tool_schema" and recent_counts[fragment.fragment_id] > 1:
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
