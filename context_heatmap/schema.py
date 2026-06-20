"""JSON-friendly схемы анализатора тепловой карты контекста."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class SessionEvent:
    """Нормализованное событие из trace harness."""

    event_id: str
    session_id: str
    turn_id: int
    timestamp: float | None
    event_type: str
    actor: str
    payload: JsonDict
    raw_ref: JsonDict
    confidence: float = 1.0

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для JSONL."""

        return asdict(self)


@dataclass
class ContextFragmentRecord:
    """Фрагмент, который был или мог быть частью prompt-пакета."""

    fragment_id: str
    session_id: str
    source_type: str
    source_name: str
    tokens: int
    token_count_method: str
    trust: str = "unknown"
    taint: str = "none"
    validity: str = "unknown"
    authority_level: str = "unknown"
    context_layer: str = "unknown"
    evictability: str = "normal"
    stability: str = "unknown"
    applicability: str = "unknown"
    normative_role: str = "none"
    goal_role: str = "none"
    target_paths: list[str] = field(default_factory=list)
    created_by_event_id: str = ""
    content_hash: str = ""
    content_excerpt_redacted: str = ""
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для JSONL."""

        return asdict(self)


@dataclass
class PacketFragment:
    """Позиция фрагмента внутри одного запроса модели."""

    fragment_id: str
    position_start: int
    position_end: int
    tokens: int
    source_type: str
    authority_level: str = "unknown"
    context_layer: str = "unknown"
    evictability: str = "normal"
    stability: str = "unknown"
    applicability: str = "unknown"
    normative_role: str = "none"
    goal_role: str = "none"

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для вложенного JSON."""

        return asdict(self)


@dataclass
class ContextPacketRecord:
    """Один prompt-пакет перед обращением к модели."""

    model_call_id: str
    session_id: str
    turn_id: int
    input_tokens: int
    context_window_tokens: int
    fragments: list[PacketFragment]
    reconstruction_confidence: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для JSONL."""

        data = asdict(self)
        data["fragments"] = [fragment.to_dict() for fragment in self.fragments]
        return data


@dataclass
class FragmentHeatRecord:
    """Оценка heat для одного фрагмента в одном prompt-пакете."""

    session_id: str
    model_call_id: str
    fragment_id: str
    heat: float
    confidence: float
    axes: JsonDict
    reasons: list[str]
    context_layer: str = "unknown"
    authority_level: str = "unknown"
    ordinary_cost: float = 0.0
    protected_status: float = 0.0
    excluded_from_red_token_share: bool = False
    protected_reasons: list[str] = field(default_factory=list)
    color: str = "green"
    evidence_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для JSONL."""

        return asdict(self)


@dataclass
class TurnHeatRecord:
    """Агрегаты нагрева на уровне обращения к модели."""

    session_id: str
    model_call_id: str
    turn_id: int
    red_token_share: float
    stale_token_share: float
    raw_tool_share: float
    # Доля окна, занятая накопленной историей ответов ассистента. Растёт,
    # когда summarization не сворачивает assistant_message — отдельный сигнал
    # window_pressure, не покрывается red_token_share.
    assistant_share: float
    active_path_purity: float
    evidence_density: float
    cold_gap_score: float
    # Сила сигнала window_pressure на этом ходе (максимум score по находкам
    # kind="window_pressure", попавшим в ход). 0 — накопления не зафиксировано.
    window_pressure_score: float
    positioned_evidence_score: float
    growth_slope: float
    taint_exposure: float
    fixed_instruction_cost: float
    goal_anchor_cost: float
    normative_status: float
    goal_status: float
    instruction_conflict_score: float
    instruction_staleness_score: float
    instruction_duplication_score: float
    instruction_scope_score: float
    instruction_integrity_score: float
    instruction_taint_score: float
    goal_integrity_score: float
    goal_supersession_score: float
    goal_conflict_score: float
    goal_overhang_score: float
    goal_cold_gap_score: float
    attached_data_taint_score: float
    top_reasons: list[str]

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для JSONL."""

        return asdict(self)


@dataclass
class Finding:
    """Человечески читаемое предупреждение с причиной и рекомендацией."""

    finding_id: str
    session_id: str
    turn_id: int
    severity: str
    kind: str
    title: str
    explanation: str
    fragment_ids: list[str]
    event_ids: list[str]
    recommendation: str
    confidence: float
    scores: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Возвращаем форму для JSONL."""

        return asdict(self)


@dataclass
class AnalysisResult:
    """Полный результат анализа одной сессии."""

    session_id: str
    events: list[SessionEvent]
    fragments: list[ContextFragmentRecord]
    packets: list[ContextPacketRecord]
    fragment_heat: list[FragmentHeatRecord]
    turn_heat: list[TurnHeatRecord]
    findings: list[Finding]
    warnings: list[JsonDict]
    session_report: JsonDict
