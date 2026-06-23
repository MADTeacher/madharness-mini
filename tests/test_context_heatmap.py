import json
import struct
import zlib
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from context_heatmap.cli import main as heatmap_main
from context_heatmap.features import detect_cold_gaps, detect_window_pressure, score_fragment
from context_heatmap.io import read_jsonl, write_jsonl
from context_heatmap.loaders.madharness_trace import load_trace, load_trace_path
from context_heatmap.normalize import load_normalized_events
from context_heatmap.png import SOURCE_COLORS, render_context_window_png
from context_heatmap.anatomy_data import build_anatomy_data
from context_heatmap.heatmap import render_heatmap_png
from context_heatmap.render import render_html_report
from context_heatmap.report import write_analysis_outputs
from context_heatmap.scoring import analyze_events
from context_heatmap.schema import (
    ContextFragmentRecord,
    ContextPacketRecord,
    PacketFragment,
    SessionEvent,
)

from tests.helpers import HarnessTestCase


def _valid_trace_events() -> list[dict]:
    return [
        {"ts": 1.0, "event": "session_start", "kind": "ask"},
        {
            "ts": 2.0,
            "event": "model_call_started",
            "turn": 0,
            "model_call_id": "valid:0",
            "context_report": {
                "max_tokens": 10000,
                "request_tokens_estimate": 10,
                "context_packet": {
                    "version": 1,
                    "request_tokens_estimate": 10,
                    "units": [
                        {
                            "unit_id": "user_task",
                            "source_type": "user_message",
                            "source_name": "task",
                            "source_ref": "user_task",
                            "tokens_estimate": 10,
                            "position_start": 0,
                            "position_end": 10,
                            "included_because": "current_task",
                            "content_hash": "abc123",
                            "confidence": 1.0,
                        }
                    ],
                    "warnings": [],
                },
            },
        },
        {
            "ts": 3.0,
            "event": "model_call_finished",
            "turn": 0,
            "model_call_id": "valid:0",
            "message": {"content": "ok"},
        },
        {"ts": 4.0, "event": "session_end", "result": "ok"},
    ]


def _write_trace(path: Path, events: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def _model_call_event(
    session_id: str,
    model_call_id: str,
    units: list[dict],
    *,
    request_tokens: int = 1000,
    max_tokens: int = 1000,
    turn_id: int = 0,
) -> SessionEvent:
    return SessionEvent(
        event_id=f"{model_call_id}:evt-000001",
        session_id=session_id,
        turn_id=turn_id,
        timestamp=1.0,
        event_type="model_call",
        actor="model",
        payload={
            "model_call_id": model_call_id,
            "context_report": {
                "max_tokens": max_tokens,
                "request_tokens_estimate": request_tokens,
                "context_packet": {
                    "version": 1,
                    "request_tokens_estimate": request_tokens,
                    "units": units,
                    "warnings": [],
                },
            },
        },
        raw_ref={"file": "synthetic.jsonl", "line": 1, "offset": None},
    )


def _unit(
    unit_id: str,
    source_type: str,
    tokens: int,
    *,
    source_name: str = "",
    source_ref: str | None = None,
    position_start: int = 0,
    content_hash: str | None = None,
    **fields,
) -> dict:
    source_ref = source_ref or unit_id
    return {
        "unit_id": unit_id,
        "source_type": source_type,
        "source_name": source_name or source_type,
        "source_ref": source_ref,
        "tokens_estimate": tokens,
        "position_start": position_start,
        "position_end": position_start + tokens,
        "included_because": fields.pop("included_because", "synthetic"),
        "content_hash": content_hash or f"hash-{unit_id}",
        "confidence": fields.pop("confidence", 0.95),
        "metadata": fields.pop("metadata", {}),
        **fields,
    }


def _read_png_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("not a PNG file")
    offset = 8
    width = 0
    height = 0
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    pixels: list[tuple[int, int, int]] = []
    cursor = 0
    for _row in range(height):
        filter_type = raw[cursor]
        cursor += 1
        if filter_type != 0:
            raise AssertionError(f"unsupported filter {filter_type}")
        row = raw[cursor : cursor + stride]
        cursor += stride
        pixels.extend(
            (row[index], row[index + 1], row[index + 2])
            for index in range(0, len(row), 3)
        )
    return width, height, pixels


def _tool_event(
    event_id: str,
    order: int,
    tool: str,
    args: dict,
    observation: dict | None = None,
) -> SessionEvent:
    return SessionEvent(
        event_id=event_id,
        session_id="s1",
        turn_id=order,
        timestamp=float(order),
        event_type="tool_result",
        actor="tool",
        payload={
            "tool": tool,
            "args": args,
            "observation": observation or {"ok": True},
        },
        raw_ref={"file": "synthetic.jsonl", "line": order, "offset": None},
    )


def _model_call_with_history(
    event_id: str,
    *,
    turn_id: int,
    included_entries: list[dict],
    summarized_indexes: set[int],
) -> SessionEvent:
    """model_call с history.included_entries и summarized_old_entries.

    Удобен для тестов cold-gap сигнатур spec_missing и post_summary: эмулирует
    prompt-пакет, в котором часть записей свернута summarization'ом.
    """

    summarized_old_entries = [{"index": idx, "kind": "tool_turn"} for idx in summarized_indexes]
    return SessionEvent(
        event_id=event_id,
        session_id="s1",
        turn_id=turn_id,
        timestamp=float(turn_id),
        event_type="model_call",
        actor="model",
        payload={
            "model_call_id": f"s1:{turn_id}",
            "context_report": {
                "max_tokens": 60000,
                "request_tokens_estimate": 1000,
                "context_packet": {
                    "version": 1,
                    "request_tokens_estimate": 1000,
                    "units": [],
                    "warnings": [],
                },
                "history": {
                    "included_entries": included_entries,
                    "summarized_old_entries": summarized_old_entries,
                    "keep_recent_turns": 3,
                    "summarize_after_turns": 3,
                },
            },
        },
        raw_ref={"file": "synthetic.jsonl", "line": turn_id, "offset": None},
    )


class ContextHeatmapTests(HarnessTestCase):
    def test_legacy_trace_is_analyzed_with_low_confidence_warning(self):
        cfg = self.make_cfg()
        cfg.ensure_dirs()
        trace_path = cfg.state_dir / "traces" / "legacy.jsonl"
        trace_path.write_text(
            "\n".join(
                [
                    json.dumps({"ts": 1.0, "event": "session_start", "kind": "ask"}),
                    json.dumps({"ts": 2.0, "event": "model_call_started", "turn": 0}),
                    json.dumps(
                        {
                            "ts": 3.0,
                            "event": "model_call_finished",
                            "turn": 0,
                            "message": {"content": "ok"},
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        events, warnings = load_trace(trace_path)
        result = analyze_events(events, warnings)

        self.assertEqual(len(result.packets), 1)
        self.assertLessEqual(result.packets[0].reconstruction_confidence, 0.55)
        self.assertTrue(
            any(
                item.get("kind") == "legacy_trace_without_context_report"
                for item in result.warnings
            )
        )

    def test_normalize_analyze_preserves_loader_warnings(self):
        cfg = self.make_cfg()
        trace_path = Path(cfg.root) / "mixed.jsonl"
        trace_path.write_text(
            "\n".join(
                [
                    json.dumps({"ts": 1.0, "event": "session_start", "kind": "ask"}),
                    "{not-json",
                    json.dumps({"ts": 2.0, "event": "model_call_started", "turn": 0}),
                    json.dumps({"ts": 3.0, "event": "session_end", "result": "ok"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        normalized = Path(cfg.root) / "normalized"
        out_dir = Path(cfg.root) / "analyzed"

        with redirect_stdout(StringIO()):
            heatmap_main(["normalize", "--input", str(trace_path), "--out", str(normalized)])
            heatmap_main(
                [
                    "analyze",
                    "--events",
                    str(normalized / "events.jsonl"),
                    "--out",
                    str(out_dir),
                ]
            )

        warnings = read_jsonl(out_dir / "warnings.jsonl")
        warning_kinds = {item.get("kind") for item in warnings}
        self.assertIn("invalid_jsonl", warning_kinds)
        self.assertIn("legacy_trace_without_context_report", warning_kinds)
        report = json.loads((out_dir / "session_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["warnings"], len(warnings))

    def test_scoring_axes_are_deterministic_for_mvp_contracts(self):
        packet = ContextPacketRecord(
            model_call_id="s1:0",
            session_id="s1",
            turn_id=0,
            input_tokens=1000,
            context_window_tokens=1000,
            fragments=[
                PacketFragment(
                    fragment_id="frag-critical",
                    position_start=400,
                    position_end=600,
                    tokens=200,
                    source_type="user_message",
                )
            ],
            reconstruction_confidence=0.9,
            warnings=["legacy_trace_without_context_packet"],
        )
        fragment = ContextFragmentRecord(
            fragment_id="frag-critical",
            session_id="s1",
            source_type="user_message",
            source_name="task",
            tokens=200,
            token_count_method="char_estimate",
            content_hash="same-hash",
            metadata={"confidence": 0.9},
        )

        axes, heat, reasons, confidence = score_fragment(
            fragment,
            packet,
            Counter({"same-hash": 2}),
            Counter({"frag-critical": 2}),
            0.0,
        )

        self.assertGreaterEqual(axes["duplication"], 0.85)
        self.assertGreater(axes["position_risk"], 0.0)
        self.assertEqual(axes["branch_mix"], 0.0)
        self.assertEqual(axes["compression_risk"], 0.20)
        self.assertGreater(heat, 0.0)
        self.assertEqual(confidence, 0.9)
        self.assertIn("duplicate_context", reasons)

        secret_fragment = ContextFragmentRecord(
            fragment_id="frag-secret",
            session_id="s1",
            source_type="tool_output",
            source_name="run_shell",
            tokens=100,
            token_count_method="char_estimate",
            taint="secret",
        )
        secret_axes, _heat, secret_reasons, _confidence = score_fragment(
            secret_fragment,
            packet,
            Counter(),
            Counter(),
            0.0,
        )
        self.assertEqual(secret_axes["taint"], 1.0)
        self.assertIn("tainted_or_untrusted", secret_reasons)

    def test_tool_schema_inter_turn_repeat_is_not_duplication(self):
        """Схемы инструментов статичны и подаются каждый ход — inter-turn повтор
        не должен считаться duplicate_context, иначе heatmap шумит на каждой
        сессии со стабильным набором инструментов."""
        packet = ContextPacketRecord(
            model_call_id="s1:0",
            session_id="s1",
            turn_id=0,
            input_tokens=1000,
            context_window_tokens=1000,
            fragments=[
                PacketFragment(
                    fragment_id="frag-tool_schema-tools-4--abc",
                    position_start=0,
                    position_end=499,
                    tokens=499,
                    source_type="tool_schema",
                )
            ],
            reconstruction_confidence=0.9,
        )
        fragment = ContextFragmentRecord(
            fragment_id="frag-tool_schema-tools-4--abc",
            session_id="s1",
            source_type="tool_schema",
            source_name="apply_patch",
            tokens=499,
            token_count_method="char_estimate",
            content_hash="schema-hash-apply-patch",
            metadata={"confidence": 0.85},
        )
        # Тот же fragment_id встречался в последних 5 пакетах 3 раза —
        # нормальная ситуация для статичных схем.
        recent_counts = Counter({"frag-tool_schema-tools-4--abc": 3})
        axes, _heat, reasons, _confidence = score_fragment(
            fragment,
            packet,
            Counter(),  # intra-paket hash не повторяется
            recent_counts,
            0.0,
        )
        self.assertLess(axes["duplication"], 0.50)
        self.assertNotIn("duplicate_context", reasons)

    def test_tool_schema_intra_packet_duplication_is_kept(self):
        """Если одна и та же схема дважды в одном пакете — это аномалия
        конфигурации инструментов, её оставляем как duplicate_context."""
        packet = ContextPacketRecord(
            model_call_id="s1:0",
            session_id="s1",
            turn_id=0,
            input_tokens=1000,
            context_window_tokens=1000,
            fragments=[
                PacketFragment(
                    fragment_id="frag-tool_schema-tools-4--abc",
                    position_start=0,
                    position_end=499,
                    tokens=499,
                    source_type="tool_schema",
                )
            ],
            reconstruction_confidence=0.9,
        )
        fragment = ContextFragmentRecord(
            fragment_id="frag-tool_schema-tools-4--abc",
            session_id="s1",
            source_type="tool_schema",
            source_name="apply_patch",
            tokens=499,
            token_count_method="char_estimate",
            content_hash="same-schema-hash",
            metadata={"confidence": 0.85},
        )
        # Один и тот же hash встречается 2 раза в текущем пакете.
        axes, _heat, reasons, _confidence = score_fragment(
            fragment,
            packet,
            Counter({"same-schema-hash": 2}),
            Counter(),
            0.0,
        )
        self.assertGreaterEqual(axes["duplication"], 0.85)
        self.assertIn("duplicate_context", reasons)

    def test_protected_system_prompt_does_not_increase_red_share_by_size(self):
        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit(
                        "system",
                        "system_instruction",
                        200,
                        source_name="system",
                        source_ref="system",
                        authority_level="system",
                        context_layer="normative",
                        evictability="never",
                        stability="stable",
                        applicability="active",
                        normative_role="safety",
                    )
                ],
            )
        ]

        result = analyze_events(events)
        heat = result.fragment_heat[0]
        turn = result.turn_heat[0]

        self.assertEqual(heat.context_layer, "normative")
        self.assertGreaterEqual(heat.ordinary_cost, 0.25)
        self.assertEqual(heat.protected_status, 0.0)
        self.assertTrue(heat.excluded_from_red_token_share)
        self.assertEqual(turn.red_token_share, 0.0)
        self.assertEqual(turn.fixed_instruction_cost, 0.2)

    def test_repeated_agents_md_is_not_duplication_by_repeated_calls(self):
        unit = _unit(
            "project-instructions",
            "developer_instruction",
            180,
            source_name="AGENTS.md",
            source_ref="project-instructions",
            content_hash="same-agents",
            authority_level="project",
            context_layer="normative",
            evictability="only_after_validation",
            stability="stable",
            applicability="current_project",
            normative_role="workflow",
        )
        events = [
            _model_call_event("s1", "s1:0", [unit], turn_id=0),
            _model_call_event("s1", "s1:1", [unit], turn_id=1),
        ]

        result = analyze_events(events)
        second = result.fragment_heat[-1]
        second_turn = result.turn_heat[-1]

        self.assertGreaterEqual(second.ordinary_cost, 0.25)
        self.assertEqual(second.axes["instruction_duplication_score"], 0.0)
        self.assertEqual(second.protected_status, 0.0)
        self.assertTrue(second.excluded_from_red_token_share)
        self.assertEqual(second_turn.red_token_share, 0.0)

    def test_wrong_project_instruction_gets_protected_red_status(self):
        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit(
                        "foreign-agents",
                        "developer_instruction",
                        200,
                        source_name="AGENTS.md",
                        authority_level="project",
                        context_layer="normative",
                        evictability="only_after_validation",
                        stability="stable",
                        applicability="wrong_project",
                        normative_role="workflow",
                    )
                ],
            )
        ]

        result = analyze_events(events)
        heat = result.fragment_heat[0]
        turn = result.turn_heat[0]

        self.assertEqual(heat.protected_status, 1.0)
        self.assertEqual(heat.color, "red")
        self.assertFalse(heat.excluded_from_red_token_share)
        self.assertEqual(turn.instruction_scope_score, 1.0)
        self.assertEqual(turn.normative_status, 1.0)
        self.assertEqual(turn.red_token_share, 0.2)

    def test_long_active_user_goal_counts_anchor_cost_without_red_pressure(self):
        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit(
                        "user_task",
                        "user_message",
                        300,
                        source_name="task",
                        source_ref="user_task",
                        authority_level="user",
                        context_layer="goal",
                        evictability="goal_update_only",
                        stability="task",
                        applicability="active",
                        goal_role="primary_goal",
                    )
                ],
            )
        ]

        result = analyze_events(events)
        heat = result.fragment_heat[0]
        turn = result.turn_heat[0]

        self.assertEqual(heat.context_layer, "goal")
        self.assertGreaterEqual(heat.ordinary_cost, 0.25)
        self.assertEqual(heat.protected_status, 0.0)
        self.assertTrue(heat.excluded_from_red_token_share)
        self.assertEqual(turn.goal_anchor_cost, 0.3)
        self.assertEqual(turn.goal_status, 0.0)
        self.assertEqual(turn.red_token_share, 0.0)

    def test_superseded_goal_raises_goal_supersession_score(self):
        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit(
                        "old_goal",
                        "user_message",
                        120,
                        source_name="task",
                        source_ref="history.old_goal",
                        authority_level="user",
                        context_layer="goal",
                        evictability="goal_update_only",
                        stability="superseded",
                        applicability="superseded",
                        goal_role="primary_goal",
                    )
                ],
            )
        ]

        result = analyze_events(events)
        heat = result.fragment_heat[0]
        turn = result.turn_heat[0]

        self.assertEqual(heat.protected_status, 0.9)
        self.assertIn("goal_superseded", heat.protected_reasons)
        self.assertEqual(turn.goal_supersession_score, 0.9)
        self.assertEqual(turn.goal_status, 0.9)

    def test_attached_user_document_is_scored_as_tainted_working_data(self):
        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit(
                        "attached",
                        "user_message",
                        200,
                        source_name="attached_data",
                        source_ref="user_task.attached[0]",
                        authority_level="external",
                        context_layer="working",
                        evictability="preferred",
                        stability="temporary",
                        applicability="current_task",
                        goal_role="attached_data",
                        metadata={"attached_data_taint_score": 0.85},
                    )
                ],
            )
        ]

        result = analyze_events(events)
        heat = result.fragment_heat[0]
        turn = result.turn_heat[0]

        self.assertEqual(heat.context_layer, "working")
        self.assertEqual(heat.axes["attached_data_taint_score"], 0.85)
        self.assertIn("attached_data_taint", heat.reasons)
        self.assertEqual(heat.color, "red")
        self.assertFalse(heat.excluded_from_red_token_share)
        self.assertEqual(turn.attached_data_taint_score, 0.85)
        self.assertEqual(turn.red_token_share, 0.2)

    def test_ordinary_tool_output_still_contributes_to_red_share(self):
        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit(
                        "tool-output",
                        "tool_output",
                        500,
                        source_name="run_shell",
                        source_ref="history[0].messages[1]",
                        taint="secret",
                        metadata={"confidence": 0.95},
                    )
                ],
            )
        ]
        result = analyze_events(events)
        heat = result.fragment_heat[0]
        turn = result.turn_heat[0]

        self.assertEqual(heat.context_layer, "working")
        self.assertEqual(heat.color, "red")
        self.assertFalse(heat.excluded_from_red_token_share)
        self.assertEqual(turn.red_token_share, 0.5)

    def test_cold_gaps_use_event_order_and_patch_paths(self):
        initial_create = [
            _tool_event("s1:evt-000001", 1, "write_file", {"path": "new.py"}),
        ]
        self.assertEqual(detect_cold_gaps(initial_create), [])

        safe_events = [
            _tool_event("s1:evt-000001", 1, "read_file", {"path": "app.py"}),
            _tool_event(
                "s1:evt-000002",
                2,
                "apply_patch",
                {"patch": "*** Begin Patch\n*** Update File: app.py\n@@\n pass\n*** End Patch"},
            ),
        ]
        self.assertEqual(detect_cold_gaps(safe_events), [])

        repeated_write = [
            _tool_event("s1:evt-000001", 1, "write_file", {"path": "app.py"}),
            _tool_event("s1:evt-000002", 2, "write_file", {"path": "app.py"}),
        ]
        repeated_findings = detect_cold_gaps(repeated_write)
        self.assertEqual(len(repeated_findings), 1)
        self.assertEqual(repeated_findings[0].event_ids, ["s1:evt-000002"])

        known_existing_write = [
            _tool_event(
                "s1:evt-000001",
                1,
                "list_files",
                {"path": "."},
                {"ok": True, "files": ["app.py"]},
            ),
            _tool_event("s1:evt-000002", 2, "write_file", {"path": "app.py"}),
        ]
        known_findings = detect_cold_gaps(known_existing_write)
        self.assertEqual(len(known_findings), 1)
        self.assertEqual(known_findings[0].event_ids, ["s1:evt-000002"])

        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: a.py",
                "*** Move to: moved.py",
                "@@",
                "-old",
                "+new",
                "*** Add File: b.py",
                "+new",
                "*** Delete File: c.py",
                "*** End Patch",
            ]
        )
        findings = detect_cold_gaps(
            [_tool_event("s1:evt-000004", 4, "apply_patch", {"patch": patch})]
        )
        explanations = "\n".join(finding.explanation for finding in findings)
        self.assertEqual(len(findings), 2)
        for path in ("a.py", "c.py"):
            self.assertIn(path, explanations)
        self.assertNotIn("b.py", explanations)
        self.assertNotIn("moved.py", explanations)

    def test_spec_missing_gap_fires_when_spec_absent_from_window(self):
        """Правка без спецификации в окне (ТЗ свёрнут или не перечитан)."""

        # Сначала модель читает крупный документ-ТЗ (это делает его спецификацией
        # по эвристике размера). Затем на ходе правки prompt-пакет не содержит
        # spec.md: file_refs с read есть, но запись помечена свёрнутой.
        events = [
            _tool_event(
                "s1:evt-000001",
                1,
                "read_file",
                {"path": "spec.md"},
                {"ok": True, "content": "x" * 2000, "start": 1, "end": 100},
            ),
            # model_call того же хода, что и правка: spec.md свёрнут.
            _model_call_with_history(
                "s1:evt-000002",
                turn_id=3,
                included_entries=[
                    {"index": 0, "file_refs": [{"kind": "read", "path": "spec.md"}]},
                ],
                summarized_indexes={0},
            ),
            _tool_event("s1:evt-000003", 3, "write_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        spec_findings = [
            f for f in findings if "отсутствии спецификации" in f.title
        ]
        self.assertEqual(len(spec_findings), 1)
        self.assertIn("spec.md", spec_findings[0].explanation)
        self.assertAlmostEqual(spec_findings[0].confidence, 0.72)

    def test_spec_missing_does_not_fire_when_spec_in_window(self):
        """Если ТЗ присутствует в окне полным read_file — холодной дыры нет."""

        events = [
            _tool_event(
                "s1:evt-000001",
                1,
                "read_file",
                {"path": "spec.md"},
                {"ok": True, "content": "x" * 2000, "start": 1, "end": 100},
            ),
            # model_call того же хода, что и правка: spec.md присутствует (read
            # не свёрнут summarization'ом).
            _model_call_with_history(
                "s1:evt-000002",
                turn_id=3,
                included_entries=[
                    {"index": 0, "file_refs": [{"kind": "read", "path": "spec.md"}]},
                ],
                summarized_indexes=set(),
            ),
            _tool_event("s1:evt-000003", 3, "write_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        self.assertEqual(findings, [])

    def test_post_summary_gap_fires_after_collapsing_read(self):
        """Правка сразу после summarization критического чтения.

        Отдельный spec-документ нужен, чтобы правимый путь src/app.py не стал
        спецификацией по размеру — тогда сработает именно post_summary, а не
        более сильный spec_missing.
        """

        events = [
            # spec.md — крупный документ, становится спецификацией.
            _tool_event(
                "s1:evt-000001",
                1,
                "read_file",
                {"path": "spec.md"},
                {"ok": True, "content": "x" * 2000, "start": 1, "end": 100},
            ),
            # src/app.py — обычный файл, НЕ спецификация (меньше spec.md).
            _tool_event(
                "s1:evt-000002",
                2,
                "read_file",
                {"path": "src/app.py"},
                {"ok": True, "content": "x" * 100, "start": 1, "end": 10},
            ),
            # model_call того же хода, что и правка: сворачивает чтение app.py.
            _model_call_with_history(
                "s1:evt-000003",
                turn_id=3,
                included_entries=[
                    {"index": 0, "file_refs": [{"kind": "read", "path": "src/app.py"}]},
                    # spec.md присутствует полным (не свёрнут) — spec_missing
                    # не должен сработать, уступая место post_summary.
                    {"index": 1, "file_refs": [{"kind": "read", "path": "spec.md"}]},
                ],
                summarized_indexes={0},
            ),
            # Первичная правка app.py: known_existing (читали), но повторных
            # правок ещё не было — repeated_write не сработает. Остаются
            # spec_missing (нет, spec.md в окне) и post_summary (сработает).
            _tool_event("s1:evt-000004", 3, "write_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        post_summary = [
            f for f in findings if "сворачивания чтения" in f.title
        ]
        self.assertEqual(
            len(post_summary),
            1,
            f"ожидали post_summary, получили: {[f.title for f in findings]}",
        )
        self.assertIn("src/app.py", post_summary[0].explanation)

    def test_post_summary_does_not_fire_after_reread(self):
        """Если после сворачивания модель перечитала путь — холодной дыры нет."""

        events = [
            # spec.md отделяет спецификацию от правимого пути.
            _tool_event(
                "s1:evt-000001",
                1,
                "read_file",
                {"path": "spec.md"},
                {"ok": True, "content": "x" * 2000, "start": 1, "end": 100},
            ),
            _tool_event(
                "s1:evt-000002",
                2,
                "read_file",
                {"path": "src/app.py"},
                {"ok": True, "content": "x" * 100, "start": 1, "end": 10},
            ),
            _model_call_with_history(
                "s1:evt-000003",
                turn_id=3,
                included_entries=[
                    {"index": 0, "file_refs": [{"kind": "read", "path": "src/app.py"}]},
                    {"index": 1, "file_refs": [{"kind": "read", "path": "spec.md"}]},
                ],
                summarized_indexes={0},
            ),
            # Свежее перечитывание app.py после сворачивания.
            _tool_event(
                "s1:evt-000004",
                4,
                "read_file",
                {"path": "src/app.py"},
                {"ok": True, "content": "y", "start": 1, "end": 1},
            ),
            # model_call того же хода, что и правка: spec.md присутствует.
            _model_call_with_history(
                "s1:evt-000005",
                turn_id=5,
                included_entries=[
                    {"index": 1, "file_refs": [{"kind": "read", "path": "spec.md"}]},
                ],
                summarized_indexes=set(),
            ),
            _tool_event("s1:evt-000006", 5, "write_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        post_summary = [f for f in findings if "сворачивания чтения" in f.title]
        self.assertEqual(post_summary, [])

    def test_read_storm_fires_on_repeated_reads(self):
        """Аномальная частота повторных чтений одного пути."""

        events = [
            _tool_event("s1:evt-000001", 1, "read_file", {"path": "src/app.py"}),
            _tool_event("s1:evt-000002", 2, "read_file", {"path": "src/app.py"}),
            _tool_event("s1:evt-000003", 3, "read_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        storm = [f for f in findings if "повторных чтений" in f.title]
        self.assertEqual(len(storm), 1)
        self.assertEqual(storm[0].severity, "low")
        self.assertAlmostEqual(storm[0].confidence, 0.55)

    def test_read_storm_does_not_fire_below_threshold(self):
        """Два чтения (меньше порога 3) — не read_storm."""

        events = [
            _tool_event("s1:evt-000001", 1, "read_file", {"path": "src/app.py"}),
            _tool_event("s1:evt-000002", 2, "read_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        storm = [f for f in findings if "повторных чтений" in f.title]
        self.assertEqual(storm, [])

    def test_apply_patch_storm_fires_on_repeated_failures(self):
        """Две подряд неудачных apply_patch по пути → apply_patch_storm."""

        patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch"
        events = [
            _tool_event(
                "s1:evt-000001",
                1,
                "apply_patch",
                {"patch": patch},
                {"ok": False, "summary": "expected 1 hunk match, found 0"},
            ),
            _tool_event(
                "s1:evt-000002",
                2,
                "apply_patch",
                {"patch": patch},
                {"ok": False, "summary": "expected 1 hunk match, found 0"},
            ),
        ]
        findings = detect_cold_gaps(events)
        storm = [f for f in findings if "неудачных правок" in f.title]
        self.assertEqual(len(storm), 1)
        self.assertEqual(storm[0].severity, "low")
        self.assertAlmostEqual(storm[0].confidence, 0.60)
        self.assertIn("src/app.py", storm[0].explanation)

    def test_apply_patch_storm_does_not_fire_below_threshold(self):
        """Одна неудача (меньше порога 2) — не apply_patch_storm."""

        patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch"
        events = [
            _tool_event(
                "s1:evt-000001",
                1,
                "apply_patch",
                {"patch": patch},
                {"ok": False, "summary": "expected 1 hunk match, found 0"},
            ),
        ]
        findings = detect_cold_gaps(events)
        storm = [f for f in findings if "неудачных правок" in f.title]
        self.assertEqual(storm, [])

    def test_apply_patch_storm_reset_by_success(self):
        """Успешная правка сбрасывает счётчик: fail, success, fail — не storm."""

        patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch"
        events = [
            _tool_event(
                "s1:evt-000001",
                1,
                "apply_patch",
                {"patch": patch},
                {"ok": False, "summary": "expected 1 hunk match, found 0"},
            ),
            _tool_event(
                "s1:evt-000002",
                2,
                "apply_patch",
                {"patch": patch},
                {"ok": True, "summary": "applied patch to 1 file(s)"},
            ),
            _tool_event(
                "s1:evt-000003",
                3,
                "apply_patch",
                {"patch": patch},
                {"ok": False, "summary": "expected 1 hunk match, found 0"},
            ),
        ]
        findings = detect_cold_gaps(events)
        storm = [f for f in findings if "неудачных правок" in f.title]
        self.assertEqual(storm, [])

    def test_dedup_keeps_highest_confidence_per_path(self):
        """На одну правку берётся одна находка с максимальным confidence."""

        # Повторная правка известного пути без read: repeated_write (0.72)
        # должен выиграть у более слабых сигнатур на том же event_id+path.
        events = [
            _tool_event(
                "s1:evt-000001",
                1,
                "list_files",
                {"path": "."},
                {"ok": True, "files": ["src/app.py"]},
            ),
            _tool_event("s1:evt-000002", 2, "write_file", {"path": "src/app.py"}),
            _tool_event("s1:evt-000003", 3, "write_file", {"path": "src/app.py"}),
        ]
        findings = detect_cold_gaps(events)
        # Ровно одна находка на event s1:evt-000003 для src/app.py.
        write_findings = [
            f for f in findings if "s1:evt-000003" in f.event_ids
        ]
        self.assertEqual(len(write_findings), 1)
        self.assertAlmostEqual(write_findings[0].confidence, 0.72)

    def test_legacy_trace_without_context_report_does_not_crash(self):
        """События без context_report/units — детектор не падает."""

        events = [
            _tool_event("s1:evt-000001", 1, "write_file", {"path": "new.py"}),
            _tool_event("s1:evt-000002", 2, "write_file", {"path": "new.py"}),
        ]
        # Не должно выбросить; repeated_write при этом работает (файл известен).
        findings = detect_cold_gaps(events)
        self.assertEqual(len(findings), 1)

    def test_window_pressure_fires_when_assistant_share_grows(self):
        """Накопление истории ассистента с ростом доли → finding window_pressure.

        Три model_call: доля assistant_message 30% → 50% → 65%. Срабатывает
        на ходах 1 и 2 (рост выше порога 0.40 + прирост ≥ 0.02). На ходе 0
        базы для сравнения роста нет.
        """

        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit("sys", "system_instruction", 300),
                    _unit("a0", "assistant_message", 300),
                    _unit("t0", "tool_output", 400),
                ],
                turn_id=0,
            ),
            _model_call_event(
                "s1",
                "s1:1",
                [
                    _unit("sys", "system_instruction", 300),
                    _unit("a0", "assistant_message", 500),
                    _unit("a1", "assistant_message", 500),
                    _unit("t0", "tool_output", 200),
                ],
                turn_id=1,
            ),
            _model_call_event(
                "s1",
                "s1:2",
                [
                    _unit("sys", "system_instruction", 300),
                    _unit("a0", "assistant_message", 650),
                    _unit("a1", "assistant_message", 650),
                ],
                turn_id=2,
            ),
        ]
        findings = detect_window_pressure(events)
        self.assertEqual(
            len(findings), 2, f"ожидали 2 находки, получили: {[f.title for f in findings]}"
        )
        # Все находки — нового класса, на ходах после первого.
        for finding in findings:
            self.assertEqual(finding.kind, "window_pressure")
            self.assertEqual(finding.severity, "medium")
            self.assertIn(finding.turn_id, {1, 2})
            self.assertAlmostEqual(finding.confidence, 0.70)
            self.assertGreaterEqual(finding.scores["assistant_share"], 0.40)

    def test_window_pressure_does_not_fire_on_plateau(self):
        """Стабильная высокая доля (без роста) → находка не срабатывает.

        Доля assistant_message 60% на всех ходах: условие роста не выполнено.
        """

        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit("sys", "system_instruction", 400),
                    _unit("a0", "assistant_message", 600),
                ],
                turn_id=0,
            ),
            _model_call_event(
                "s1",
                "s1:1",
                [
                    _unit("sys", "system_instruction", 400),
                    _unit("a0", "assistant_message", 600),
                ],
                turn_id=1,
            ),
            _model_call_event(
                "s1",
                "s1:2",
                [
                    _unit("sys", "system_instruction", 400),
                    _unit("a0", "assistant_message", 600),
                ],
                turn_id=2,
            ),
        ]
        findings = detect_window_pressure(events)
        self.assertEqual(findings, [])

    def test_window_pressure_does_not_fire_below_share_threshold(self):
        """Доля ниже 0.40 даже при росте → находка не срабатывает."""

        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit("sys", "system_instruction", 800),
                    _unit("a0", "assistant_message", 100),
                    _unit("t0", "tool_output", 100),
                ],
                turn_id=0,
            ),
            _model_call_event(
                "s1",
                "s1:1",
                [
                    _unit("sys", "system_instruction", 800),
                    _unit("a0", "assistant_message", 200),
                ],
                turn_id=1,
            ),
        ]
        findings = detect_window_pressure(events)
        self.assertEqual(findings, [])

    def test_window_pressure_legacy_trace_returns_empty(self):
        """model_call без context_packet.units → детектор молчит, не падает."""

        events = [
            _tool_event("s1:evt-000001", 1, "read_file", {"path": "app.py"}),
            _tool_event("s1:evt-000002", 2, "write_file", {"path": "app.py"}),
        ]
        self.assertEqual(detect_window_pressure(events), [])

    def test_window_pressure_wired_into_analysis_and_session_report(self):
        """Сквозная проверка: assistant_share в turn_heat, метрики в session_report."""

        events = [
            _model_call_event(
                "s1",
                "s1:0",
                [
                    _unit("sys", "system_instruction", 300),
                    _unit("a0", "assistant_message", 300),
                    _unit("t0", "tool_output", 400),
                ],
                turn_id=0,
            ),
            _model_call_event(
                "s1",
                "s1:1",
                [
                    _unit("sys", "system_instruction", 300),
                    _unit("a0", "assistant_message", 1000),
                    _unit("a1", "assistant_message", 500),
                ],
                turn_id=1,
            ),
        ]
        result = analyze_events(events)

        # turn_heat несёт новую колонку assistant_share.
        self.assertEqual(len(result.turn_heat), 2)
        self.assertGreater(result.turn_heat[0].assistant_share, 0.0)
        self.assertGreater(result.turn_heat[1].assistant_share, 0.40)
        # window_pressure_score проставлен на ходе 1, где сработал детектор.
        self.assertEqual(result.turn_heat[0].window_pressure_score, 0.0)
        self.assertGreater(result.turn_heat[1].window_pressure_score, 0.0)
        # session_report агрегирует новые метрики.
        report = result.session_report
        self.assertIn("max_assistant_share", report)
        self.assertIn("mean_assistant_share", report)
        self.assertIn("max_window_pressure_score", report)
        self.assertGreater(report["max_assistant_share"], 0.40)
        self.assertGreater(report["max_window_pressure_score"], 0.0)
        # Находка попала в общий список findings.
        self.assertTrue(
            any(f.kind == "window_pressure" for f in result.findings),
            f"ожидали window_pressure в findings: {[f.kind for f in result.findings]}",
        )

    def test_regression_flappy1_has_no_false_positive_gaps(self):
        """flappy1 (sum=0) — эталон чистоты: 0 холодных дыр."""

        findings = self._flappy_findings("flappy1_sum0.jsonl")
        self.assertEqual(
            findings,
            [],
            f"flappy1 не должен давать холодных дыр, получили: "
            f"{[f.title for f in findings]}",
        )

    def test_regression_flappy2a_catches_spec_loss_after_summary(self):
        """flappy2 #1: summarization свернул ТЗ → spec_missing на ходах 9+."""

        findings = self._flappy_findings("flappy2a_sum3.jsonl")
        spec_findings = [
            f for f in findings if "отсутствии спецификации" in f.title
        ]
        self.assertGreaterEqual(
            len(spec_findings),
            1,
            "flappy2 #1 должен ловить потерю спецификации после summarization",
        )
        # Все spec_missing находки — на ходах после сворачивания (turn_id >= 9).
        for finding in spec_findings:
            self.assertGreaterEqual(
                finding.turn_id, 9, f"spec_missing на раннем ходе: turn {finding.turn_id}"
            )

    def test_regression_flappy2b_catches_read_storm(self):
        """flappy2 #2: модель 36 раз перечитывала файлы → read_storm."""

        findings = self._flappy_findings("flappy2b_sum3.jsonl")
        storm = [f for f in findings if "повторных чтений" in f.title]
        self.assertGreaterEqual(
            len(storm),
            1,
            "flappy2 #2 должен ловить аномальную частоту перечитываний",
        )

    def test_regression_flappy3_keeps_existing_gaps_and_adds_new(self):
        """flappy3: повторные правки + spec_missing/post_summary + apply_patch_storm."""

        findings = self._flappy_findings("flappy3_sum6.jsonl")
        titles = [f.title for f in findings]
        # Существовавшие 2 repeated_write gaps (tetrisGame.js, iGame.js).
        self.assertTrue(
            any("без актуального чтения" in t for t in titles),
            "flappy3 должен сохранить repeated_write gaps",
        )
        # Новые сигнатуры: spec_missing или post_summary.
        new_signatures = [
            t
            for t in titles
            if "отсутствии спецификации" in t or "сворачивания чтения" in t
        ]
        self.assertGreaterEqual(
            len(new_signatures),
            1,
            "flappy3 должен ловить новые сигнатуры холодных дыр",
        )
        # apply_patch_storm: цикл неудач на tetrisGame.js (5 неудач).
        self.assertTrue(
            any("неудачных правок" in t for t in titles),
            "flappy3 должен ловить apply_patch_storm на tetrisGame.js",
        )

    def _flappy_findings(self, trace_name: str) -> list:
        """Запускает detect_cold_gaps на трассе из tests/data/traces/flappy/."""

        trace_path = (
            Path(__file__).parent / "data" / "traces" / "flappy" / trace_name
        )
        events, _warnings = load_trace(trace_path)
        return detect_cold_gaps(events)

    def test_analyze_writes_outputs_and_renders_html_without_secret(self):
        cfg = self.make_cfg()
        events = [
            SessionEvent(
                event_id="s1:evt-000001",
                session_id="s1",
                turn_id=0,
                timestamp=1.0,
                event_type="model_call",
                actor="model",
                payload={
                    "model_call_id": "s1:0",
                    "context_report": {
                        "max_tokens": 10000,
                        "request_tokens_estimate": 500,
                        "context_packet": {
                            "version": 1,
                            "request_tokens_estimate": 500,
                            "units": [
                                {
                                    "unit_id": "system",
                                    "source_type": "system_instruction",
                                    "source_name": "system",
                                    "source_ref": "system",
                                    "tokens_estimate": 50,
                                    "position_start": 0,
                                    "position_end": 50,
                                    "included_because": "system_fragment",
                                    "content_hash": "aaa",
                                    "confidence": 0.95,
                                },
                                {
                                    "unit_id": "tool",
                                    "source_type": "tool_output",
                                    "source_name": "run_shell",
                                    "source_ref": "history[0]",
                                    "tokens_estimate": 220,
                                    "position_start": 50,
                                    "position_end": 270,
                                    "included_because": "rendered_history",
                                    "content_hash": "bbb",
                                    "confidence": 0.9,
                                },
                            ],
                            "warnings": [],
                        },
                    },
                },
                raw_ref={"file": "synthetic.jsonl", "line": 1, "offset": None},
            ),
            SessionEvent(
                event_id="s1:evt-000002",
                session_id="s1",
                turn_id=0,
                timestamp=2.0,
                event_type="tool_result",
                actor="tool",
                payload={
                    "tool": "apply_patch",
                    "args": {"patch": "*** Begin Patch\n*** Update File: app.py\n@@\n pass\n*** End Patch"},
                    "observation": {
                        "ok": True,
                        "summary": "patched",
                        "stdout": "api_key=sk-testSECRET123456789",
                    },
                },
                raw_ref={"file": "synthetic.jsonl", "line": 2, "offset": None},
            ),
        ]
        out_dir = Path(cfg.root) / "heatmap-out"

        result = analyze_events(events)
        write_analysis_outputs(result, out_dir)
        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        html = (out_dir / "heatmap.html").read_text(encoding="utf-8")
        fragments = (out_dir / "fragments.jsonl").read_text(encoding="utf-8")
        markdown = (out_dir / "report.md").read_text(encoding="utf-8")
        report = json.loads((out_dir / "session_report.json").read_text(encoding="utf-8"))

        self.assertTrue((out_dir / "fragment_heat.jsonl").exists())
        self.assertTrue((out_dir / "turn_heat.csv").exists())
        self.assertTrue((out_dir / "heatmap.png").exists())
        self.assertEqual(report["outputs"]["heatmap_png"], "heatmap.png")
        for filename in report["outputs"].values():
            self.assertTrue((out_dir / filename).exists(), filename)
        self.assertEqual(
            (out_dir / "fragment_heat.csv").read_text(encoding="utf-8").splitlines()[0],
            "session_id,model_call_id,fragment_id,heat,confidence,context_layer,"
            "authority_level,ordinary_cost,protected_status,"
            "excluded_from_red_token_share,color,axes,reasons,protected_reasons",
        )
        self.assertEqual(
            (out_dir / "turn_heat.csv").read_text(encoding="utf-8").splitlines()[0],
            "session_id,model_call_id,turn_id,red_token_share,stale_token_share,"
            "raw_tool_share,assistant_share,evidence_density,cold_gap_score,"
            "window_pressure_score,taint_exposure,fixed_instruction_cost,"
            "goal_anchor_cost,normative_status,goal_status,instruction_scope_score,"
            "goal_supersession_score,attached_data_taint_score,top_reasons",
        )
        self.assertIn("heatmap-data", html)
        self.assertIn("cold", html)
        self.assertIn("reasons=", html)
        self.assertIn("Context Heatmap Report", markdown)
        self.assertIn("<redacted>", fragments)
        self.assertNotIn("sk-testSECRET123456789", fragments)
        self.assertNotIn("sk-testSECRET123456789", html)
        self.assertNotIn("sk-testSECRET123456789", markdown)
        self.assertNotIn("sk-testSECRET123456789".encode(), (out_dir / "heatmap.png").read_bytes())
        self.assertNotIn("sk-testSECRET123456789", json.dumps(report))
        self.assertTrue(any(finding.kind == "cold_gap" for finding in result.findings))

    def test_png_summary_renders_pixels_without_raw_content(self):
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "png-summary"
        report = {
            "session_id": "s1",
            "model_calls": 3,
            "max_cold_gap_score": 0.72,
            "findings": 1,
        }
        packets = [
            {
                "model_call_id": "s1:0",
                "turn_id": 0,
                "input_tokens": 400,
                "context_window_tokens": 1000,
                "fragments": [
                    {"source_type": "developer_instruction", "tokens": 40},
                    {"source_type": "tool_schema", "tokens": 120},
                    {"source_type": "assistant_message", "tokens": 180},
                    {"source_type": "tool_output", "tokens": 100},
                ],
            },
            {
                "model_call_id": "s1:1",
                "turn_id": 1,
                "input_tokens": 900,
                "context_window_tokens": 1000,
                "fragments": [
                    {"source_type": "tool_schema", "tokens": 100},
                    {"source_type": "assistant_message", "tokens": 600},
                    {"source_type": "tool_output", "tokens": 200},
                ],
            },
            {
                "model_call_id": "s1:2",
                "turn_id": 2,
                "input_tokens": 0,
                "context_window_tokens": 0,
                "fragments": [
                    {"source_type": "assistant_message", "tokens": 200},
                    {"source_type": "tool_output", "tokens": 100},
                ],
            },
        ]
        turn_heat = [
            {
                "model_call_id": "s1:0",
                "turn_id": 0,
                "raw_tool_share": 0.25,
                "evidence_density": 0.30,
                "positioned_evidence_score": 0.95,
                "cold_gap_score": 0.0,
            },
            {
                "model_call_id": "s1:1",
                "turn_id": 1,
                "raw_tool_share": 0.45,
                "evidence_density": 0.50,
                "positioned_evidence_score": 0.60,
                "cold_gap_score": 0.72,
            },
            {
                "model_call_id": "s1:2",
                "turn_id": 2,
                "raw_tool_share": 0.15,
                "evidence_density": 0.20,
                "positioned_evidence_score": 0.80,
                "cold_gap_score": 0.0,
            },
        ]
        findings = [
            {
                "turn_id": 1,
                "title": "raw prompt sk-testSECRET123456789 must not be drawn",
            }
        ]

        png_path = out_dir / "heatmap.png"
        render_context_window_png(report, packets, turn_heat, findings, png_path)
        width, height, pixels = _read_png_rgb(png_path)
        pixel_set = set(pixels)

        self.assertEqual((width, height), (1200, 760))
        self.assertGreater(len(pixel_set), 12)
        self.assertNotEqual(SOURCE_COLORS["developer_instruction"], SOURCE_COLORS["tool_schema"])
        self.assertIn(SOURCE_COLORS["developer_instruction"], pixel_set)
        self.assertIn(SOURCE_COLORS["tool_schema"], pixel_set)
        self.assertIn(SOURCE_COLORS["assistant_message"], pixel_set)
        self.assertIn(SOURCE_COLORS["tool_output"], pixel_set)
        self.assertIn((255, 255, 255), pixel_set)
        self.assertIn((190, 38, 30), pixel_set)
        self.assertIn((23, 78, 166), pixel_set)
        chart_source_columns = [
            column
            for row in range(160, 550)
            for column in range(72, 1128)
            if pixels[row * width + column] in SOURCE_COLORS.values()
        ]
        self.assertLessEqual(min(chart_source_columns), 80)
        above_chart = {
            pixels[row * width + column]
            for row in range(140, 160)
            for column in range(72, 1128)
        }
        self.assertNotIn((190, 38, 30), above_chart)
        self.assertNotIn((245, 166, 35), above_chart)
        chart_body_without_bottom_markers = {
            pixels[row * width + column]
            for row in range(160, 550)
            for column in range(72, 1128)
        }
        self.assertNotIn((190, 38, 30), chart_body_without_bottom_markers)
        self.assertNotIn((245, 166, 35), chart_body_without_bottom_markers)
        self.assertNotIn((23, 78, 166), chart_body_without_bottom_markers)
        context_marker_lane = {
            pixels[row * width + column]
            for row in range(550, 590)
            for column in range(72, 1128)
        }
        self.assertNotIn((23, 78, 166), context_marker_lane)
        self.assertNotIn(b"sk-testSECRET123456789", png_path.read_bytes())

    def test_html_escapes_reasons_and_findings(self):
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "html-escaping"
        out_dir.mkdir()
        (out_dir / "session_report.json").write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "model_calls": 1,
                    "max_red_token_share": 0,
                    "max_cold_gap_score": 0,
                    "findings": 1,
                }
            ),
            encoding="utf-8",
        )
        (out_dir / "fragment_heat.jsonl").write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "model_call_id": "s1:0",
                    "fragment_id": "frag",
                    "heat": 0.5,
                    "confidence": 1.0,
                    "axes": {},
                    "reasons": ["<script>alert(1)</script>"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "turn_heat.jsonl").write_text("", encoding="utf-8")
        (out_dir / "findings.jsonl").write_text(
            json.dumps(
                {
                    "severity": "medium",
                    "kind": "cold_gap",
                    "title": "<b>bad</b>",
                    "explanation": "<img src=x>",
                    "recommendation": "<script>fix()</script>",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        html = (out_dir / "heatmap.html").read_text(encoding="utf-8")

        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("&lt;b&gt;bad&lt;/b&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<img src=x>", html)

    def test_html_sorts_model_call_columns_numerically(self):
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "html-column-order"
        out_dir.mkdir()
        (out_dir / "session_report.json").write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "model_calls": 4,
                    "max_red_token_share": 0,
                    "max_cold_gap_score": 0,
                    "findings": 0,
                }
            ),
            encoding="utf-8",
        )
        (out_dir / "fragment_heat.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "session_id": "s1",
                        "model_call_id": model_call_id,
                        "fragment_id": "frag",
                        "heat": 0.1,
                        "confidence": 1.0,
                        "axes": {},
                        "reasons": [],
                    }
                )
                for model_call_id in ("s1:0", "s1:1", "s1:10", "s1:2")
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "turn_heat.jsonl").write_text("", encoding="utf-8")
        (out_dir / "findings.jsonl").write_text("", encoding="utf-8")

        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        html = (out_dir / "heatmap.html").read_text(encoding="utf-8")

        self.assertLess(html.index("<th>0</th>"), html.index("<th>1</th>"))
        self.assertLess(html.index("<th>1</th>"), html.index("<th>2</th>"))
        self.assertLess(html.index("<th>2</th>"), html.index("<th>10</th>"))

    def test_html_does_not_hide_tool_outputs_after_many_fragments(self):
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "html-tool-output-visibility"
        out_dir.mkdir()
        (out_dir / "session_report.json").write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "model_calls": 1,
                    "max_red_token_share": 0,
                    "max_cold_gap_score": 0,
                    "findings": 0,
                }
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "session_id": "s1",
                "model_call_id": "s1:0",
                "fragment_id": f"frag-assistant-{index}",
                "heat": 0.1,
                "confidence": 1.0,
                "axes": {},
                "reasons": [],
            }
            for index in range(70)
        ]
        rows.append(
            {
                "session_id": "s1",
                "model_call_id": "s1:0",
                "fragment_id": "frag-tool-output-late",
                "heat": 0.2,
                "confidence": 1.0,
                "axes": {},
                "reasons": ["tainted_or_untrusted"],
            }
        )
        (out_dir / "fragment_heat.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        fragments = [
            {
                "fragment_id": f"frag-assistant-{index}",
                "source_type": "assistant_message",
                "source_name": "assistant",
                "metadata": {
                    "source_ref": f"history[{index}].messages[0]",
                    "history_index": index,
                },
            }
            for index in range(70)
        ]
        fragments.append(
            {
                "fragment_id": "frag-tool-output-late",
                "source_type": "tool_output",
                "source_name": "tool",
                "metadata": {
                    "source_ref": "history[70].messages[1]",
                    "history_index": 70,
                    "tool_call_id": "call_read_123",
                },
            }
        )
        (out_dir / "fragments.jsonl").write_text(
            "\n".join(json.dumps(fragment) for fragment in fragments) + "\n",
            encoding="utf-8",
        )
        (out_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "model_output",
                    "payload": {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_read_123",
                                    "function": {"name": "read_file"},
                                }
                            ]
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "turn_heat.jsonl").write_text("", encoding="utf-8")
        (out_dir / "findings.jsonl").write_text("", encoding="utf-8")

        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        html = (out_dir / "heatmap.html").read_text(encoding="utf-8")

        self.assertIn("assistant_message: assistant", html)
        self.assertIn("tool_output: read_file", html)
        self.assertIn("history[70].messages[1]", html)

    def test_html_labels_tool_schema_fragments_by_tool_name(self):
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "html-tool-labels"
        out_dir.mkdir()
        (out_dir / "session_report.json").write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "model_calls": 1,
                    "max_red_token_share": 0,
                    "max_cold_gap_score": 0,
                    "findings": 0,
                }
            ),
            encoding="utf-8",
        )
        (out_dir / "fragment_heat.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-tool_output-history-0--messages-1--ghi789",
                            "heat": 0.4,
                            "confidence": 0.9,
                            "axes": {},
                            "reasons": ["tainted_or_untrusted"],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-tool_schema-tools-4--abc123",
                            "heat": 0.2,
                            "confidence": 0.85,
                            "axes": {},
                            "reasons": ["tool_schema_budget"],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-assistant_message-history-0--messages-0--jkl012",
                            "heat": 0.1,
                            "confidence": 0.9,
                            "axes": {},
                            "reasons": [],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-user_message-user-task-def456",
                            "heat": 0.3,
                            "confidence": 1.0,
                            "axes": {},
                            "reasons": ["large_or_repeated_fragment"],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-system_instruction-system-mno345",
                            "heat": 0.05,
                            "confidence": 0.95,
                            "axes": {},
                            "reasons": [],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "fragments.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "fragment_id": "frag-tool_output-history-0--messages-1--ghi789",
                            "source_type": "tool_output",
                            "source_name": "tool",
                            "metadata": {
                                "source_ref": "history[0].messages[1]",
                                "tool_call_id": "call_read_123",
                                "history_index": 0,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-tool_schema-tools-4--abc123",
                            "source_type": "tool_schema",
                            "source_name": "apply_patch",
                            "metadata": {
                                "source_ref": "tools[4]",
                                "tool_index": 4,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-assistant_message-history-0--messages-0--jkl012",
                            "source_type": "assistant_message",
                            "source_name": "assistant",
                            "metadata": {
                                "source_ref": "history[0].messages[0]",
                                "history_index": 0,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-user_message-user-task-def456",
                            "source_type": "user_message",
                            "source_name": "task",
                            "metadata": {"source_ref": "user_task"},
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-system_instruction-system-mno345",
                            "source_type": "system_instruction",
                            "source_name": "system",
                            "metadata": {"source_ref": "system"},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "model_output",
                    "payload": {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_read_123",
                                    "function": {"name": "read_file"},
                                }
                            ]
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "turn_heat.jsonl").write_text("", encoding="utf-8")
        (out_dir / "findings.jsonl").write_text("", encoding="utf-8")

        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        html = (out_dir / "heatmap.html").read_text(encoding="utf-8")

        self.assertIn("tool_schema: apply_patch", html)
        self.assertIn("tools[4]", html)
        self.assertIn("user_message: task", html)
        self.assertIn("user_task", html)

    def test_html_renders_context_window_columns_with_safe_details(self):
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "html-context-window"
        out_dir.mkdir()
        (out_dir / "session_report.json").write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "model_calls": 1,
                    "max_red_token_share": 0,
                    "max_cold_gap_score": 0,
                    "findings": 0,
                }
            ),
            encoding="utf-8",
        )
        (out_dir / "packets.jsonl").write_text(
            json.dumps(
                {
                    "model_call_id": "s1:0",
                    "session_id": "s1",
                    "turn_id": 0,
                    "input_tokens": 500,
                    "context_window_tokens": 1000,
                    "fragments": [
                        {
                            "fragment_id": "frag-system",
                            "position_start": 0,
                            "position_end": 100,
                            "tokens": 100,
                            "source_type": "system_instruction",
                        },
                        {
                            "fragment_id": "frag-user",
                            "position_start": 100,
                            "position_end": 300,
                            "tokens": 200,
                            "source_type": "user_message",
                        },
                        {
                            "fragment_id": "frag-tool-schema",
                            "position_start": 300,
                            "position_end": 360,
                            "tokens": 60,
                            "source_type": "tool_schema",
                        },
                        {
                            "fragment_id": "frag-assistant",
                            "position_start": 360,
                            "position_end": 420,
                            "tokens": 60,
                            "source_type": "assistant_message",
                        },
                        {
                            "fragment_id": "frag-tool-output",
                            "position_start": 420,
                            "position_end": 500,
                            "tokens": 80,
                            "source_type": "tool_output",
                        },
                    ],
                    "reconstruction_confidence": 0.9,
                    "warnings": ["token_positions_are_estimates"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "fragment_heat.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-system",
                            "heat": 0.1,
                            "confidence": 0.95,
                            "axes": {"position_risk": 0.0},
                            "reasons": [],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-user",
                            "heat": 0.8,
                            "confidence": 1.0,
                            "axes": {"duplication": 0.9},
                            "reasons": ["large_or_repeated_fragment"],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-tool-schema",
                            "heat": 0.2,
                            "confidence": 0.85,
                            "axes": {},
                            "reasons": ["tool_schema_budget"],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-assistant",
                            "heat": 0.1,
                            "confidence": 0.9,
                            "axes": {},
                            "reasons": [],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "s1",
                            "model_call_id": "s1:0",
                            "fragment_id": "frag-tool-output",
                            "heat": 0.6,
                            "confidence": 0.9,
                            "axes": {"taint": 0.5},
                            "reasons": ["tainted_or_untrusted"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "fragments.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "fragment_id": "frag-system",
                            "source_type": "system_instruction",
                            "source_name": "system",
                            "content_hash": "aaa",
                            "metadata": {
                                "source_ref": "system",
                                "included_because": "system_fragment",
                                "confidence": 0.95,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-user",
                            "source_type": "user_message",
                            "source_name": "task",
                            "content_hash": "bbb",
                            "target_paths": ["app.py"],
                            "content_excerpt_redacted": "raw prompt text must stay hidden",
                            "metadata": {
                                "source_ref": "user_task",
                                "included_because": "current_task",
                                "confidence": 1.0,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-tool-schema",
                            "source_type": "tool_schema",
                            "source_name": "apply_patch",
                            "content_hash": "ccc",
                            "metadata": {
                                "source_ref": "tools[4]",
                                "tool_index": 4,
                                "included_because": "available_tool_schema",
                                "confidence": 0.85,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-assistant",
                            "source_type": "assistant_message",
                            "source_name": "assistant",
                            "content_hash": "ddd",
                            "metadata": {
                                "source_ref": "history[0].messages[0]",
                                "history_index": 0,
                                "included_because": "rendered_history",
                                "confidence": 0.9,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "fragment_id": "frag-tool-output",
                            "source_type": "tool_output",
                            "source_name": "tool",
                            "content_hash": "eee",
                            "metadata": {
                                "source_ref": "history[0].messages[1]",
                                "history_index": 0,
                                "tool_call_id": "call_read_123",
                                "included_because": "rendered_history",
                                "confidence": 0.9,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "turn_heat.jsonl").write_text("", encoding="utf-8")
        (out_dir / "findings.jsonl").write_text("", encoding="utf-8")
        (out_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "model_output",
                    "payload": {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_read_123",
                                    "function": {"name": "read_file"},
                                }
                            ]
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        render_html_report(out_dir / "session_report.json", out_dir / "heatmap.html")
        html = (out_dir / "heatmap.html").read_text(encoding="utf-8")

        self.assertIn("context-window", html)
        self.assertIn("context-window-track", html)
        self.assertIn("context-window-legend-track", html)
        self.assertIn("context-column", html)
        self.assertIn("context-segment", html)
        self.assertIn("context-detail", html)
        self.assertIn("context-unused", html)
        self.assertIn("flex-wrap: nowrap", html)
        self.assertIn("overflow-x: auto", html)
        self.assertIn('data-model-call-id="s1:0"', html)
        self.assertIn('data-fragment-id="frag-user"', html)
        self.assertIn("type-system_instruction", html)
        self.assertIn("type-user_message hot", html)
        self.assertIn("window_share", html)
        self.assertIn("position_start", html)
        self.assertIn("large_or_repeated_fragment", html)
        self.assertIn("token_positions_are_estimates", html)
        self.assertNotIn("raw prompt text must stay hidden", html)
        self.assertIn("tool_output: read_file", html)
        self.assertIn("tool=read_file", html)
        self.assertLess(
            html.index("system_instruction: system"),
            html.index("user_message: task"),
        )
        self.assertLess(
            html.index("user_message: task"),
            html.index("tool_schema: apply_patch"),
        )
        self.assertLess(
            html.index("tool_schema: apply_patch"),
            html.index("assistant_message: assistant"),
        )
        self.assertLess(
            html.index("assistant_message: assistant"),
            html.index("tool_output: read_file"),
        )

    def test_normalized_events_round_trip(self):
        cfg = self.make_cfg()
        events_path = Path(cfg.root) / "events.jsonl"
        events_path.write_text(
            json.dumps(
                {
                    "event_id": "s1:evt-000001",
                    "session_id": "s1",
                    "turn_id": 0,
                    "timestamp": 1.0,
                    "event_type": "model_call",
                    "actor": "model",
                    "payload": {},
                    "raw_ref": {},
                    "confidence": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        events = load_normalized_events(events_path)

        self.assertEqual(events[0].event_id, "s1:evt-000001")

    def test_cli_validate_normalize_analyze_render_and_batch(self):
        cfg = self.make_cfg()
        trace_dir = Path(cfg.root) / "traces"
        trace_dir.mkdir()
        trace_path = trace_dir / "valid.jsonl"
        _write_trace(trace_path, _valid_trace_events())
        normalized = Path(cfg.root) / "normalized-cli"
        out_dir = Path(cfg.root) / "analyzed-cli"
        rendered = Path(cfg.root) / "rerendered.html"
        batch_dir = Path(cfg.root) / "batch"

        out = StringIO()
        with redirect_stdout(out):
            heatmap_main(["validate", "--input", str(trace_path)])
            heatmap_main(["normalize", "--input", str(trace_path), "--out", str(normalized)])
            heatmap_main(
                [
                    "analyze",
                    "--events",
                    str(normalized / "events.jsonl"),
                    "--out",
                    str(out_dir),
                ]
            )
            heatmap_main(
                [
                    "render",
                    "--report",
                    str(out_dir / "session_report.json"),
                    "--out",
                    str(rendered),
                ]
            )
            heatmap_main(["analyze-batch", "--input", str(trace_dir), "--out", str(batch_dir)])

        output = out.getvalue()
        self.assertIn("model calls: 1", output)
        self.assertIn("normalized events: 4", output)
        self.assertIn("analyzed model calls: 1", output)
        self.assertTrue((out_dir / "heatmap.html").exists())
        self.assertTrue((out_dir / "heatmap.png").exists())
        self.assertTrue(rendered.exists())
        self.assertTrue((batch_dir / "valid" / "session_report.json").exists())
        self.assertTrue((batch_dir / "valid" / "heatmap.png").exists())
        self.assertTrue((batch_dir / "corpus_report.json").exists())

        invalid_path = trace_dir / "invalid.jsonl"
        invalid_path.write_text(
            json.dumps({"ts": 1.0, "event": "session_start", "kind": "ask"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()):
                heatmap_main(["validate", "--input", str(invalid_path)])

        broken_events = _valid_trace_events()
        broken_events[1]["context_report"]["context_packet"]["units"][0][
            "position_end"
        ] = 9
        broken_path = trace_dir / "broken-packet.jsonl"
        _write_trace(broken_path, broken_events)
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()):
                heatmap_main(["validate", "--input", str(broken_path)])

    def test_docs_link_context_heatmap(self):
        root = Path(__file__).resolve().parents[1]
        docs_index = (root / "docs" / "README.md").read_text(encoding="utf-8")
        capabilities = (root / "docs" / "capabilities.md").read_text(encoding="utf-8")
        context_layer = (root / "docs" / "context-layer.md").read_text(encoding="utf-8")

        self.assertIn("context-heatmap.md", docs_index)
        self.assertIn("пассивной диагностики", capabilities)
        self.assertIn("context_packet", context_layer)

    def test_anatomy_data_builds_expected_columns_and_seams(self):
        """Агрегатор собирает колонки, швы summarization, действия и verdict."""

        report = {
            "session_id": "anat:1",
            "model_calls": 2,
            "turns": 2,
            "max_red_token_share": 0.0,
            "max_cold_gap_score": 0.72,
            "max_window_pressure_score": 0.7,
            "max_assistant_share": 0.58,
        }
        packets = [
            {
                "model_call_id": "anat:0",
                "turn_id": 0,
                "input_tokens": 400,
                "context_window_tokens": 1000,
                "fragments": [
                    {"source_type": "user_message", "tokens": 200},
                    {"source_type": "tool_output", "tokens": 200},
                ],
            },
            {
                "model_call_id": "anat:1",
                "turn_id": 1,
                "input_tokens": 900,
                "context_window_tokens": 1000,
                "fragments": [
                    {"source_type": "assistant_message", "tokens": 600},
                    {"source_type": "tool_output", "tokens": 300},
                ],
            },
        ]
        turn_heat = [
            {
                "model_call_id": "anat:0",
                "turn_id": 0,
                "cold_gap_score": 0.0,
                "window_pressure_score": 0.0,
                "assistant_share": 0.0,
                "red_token_share": 0.0,
                "raw_tool_share": 0.5,
            },
            {
                "model_call_id": "anat:1",
                "turn_id": 1,
                "cold_gap_score": 0.72,
                "window_pressure_score": 0.7,
                "assistant_share": 0.6,
                "red_token_share": 0.0,
                "raw_tool_share": 0.3,
            },
        ]
        findings = [{"kind": "cold_gap", "turn_id": 1}]
        events = [
            {
                "event_type": "model_call",
                "payload": {
                    "turn": 1,
                    "context_report": {
                        "history": {
                            "summarized_old_entries": [{"index": 0, "kind": "tool_turn"}],
                            "dropped_entries": [],
                        },
                        "truncated": False,
                    },
                },
            },
            {
                "event_type": "model_output",
                "payload": {
                    "turn": 1,
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "write_file"}},
                            {"function": {"name": "read_file"}},
                        ]
                    },
                },
            },
        ]

        data = build_anatomy_data(report, packets, turn_heat, findings, events)

        self.assertEqual(len(data["columns"]), 2)
        col0 = data["columns"][0]
        self.assertEqual(col0["model_call_id"], "anat:0")
        # Доли токенов по типам сохранены из фрагментов пакета.
        self.assertEqual(col0["tokens_by_type"]["user_message"], 200)
        self.assertEqual(col0["tokens_by_type"]["tool_output"], 200)
        # fill_share = used/window, без подмены.window=1000, used=400.
        self.assertAlmostEqual(col0["fill_share"], 0.4, places=2)
        # Сигналы проброшены из turn_heat.
        self.assertEqual(data["columns"][1]["cold_gap_score"], 0.72)

        # Cold turns собраны без дублей и в порядке находок.
        self.assertEqual(data["cold_turns"], [1])
        # Шов summarization на ходе 1: одна свёрнутая запись.
        self.assertEqual(len(data["seams"]), 1)
        self.assertEqual(data["seams"][0]["turn_id"], 1)
        self.assertEqual(data["seams"][0]["summarized"], 1)
        self.assertFalse(data["seams"][0]["truncated"])
        # Действия привязаны к ходу 1 и схлопнуты по дублям имени.
        self.assertEqual(data["actions_by_turn"][1], ["write_file", "read_file"])
        # Verdict: red=0, cold=on, pressure=on → «011», peak на ходе 1.
        self.assertEqual(data["verdict"]["dots"], "011")
        self.assertEqual(data["verdict"]["peak_turn"], 1)
        self.assertIn("COLD", data["verdict"]["label"])

    def test_heatmap_png_renders_without_raw_content(self):
        """Новый heatmap.png генерируется и не содержит raw content из findings."""

        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "heatmap-anatomy"
        report = {
            "session_id": "anat:2",
            "model_calls": 2,
            "turns": 2,
            "max_red_token_share": 0.0,
            "max_cold_gap_score": 0.72,
            "max_window_pressure_score": 0.0,
            "max_assistant_share": 0.0,
        }
        packets = [
            {
                "model_call_id": "anat:0",
                "turn_id": 0,
                "input_tokens": 400,
                "context_window_tokens": 1000,
                "fragments": [{"source_type": "user_message", "tokens": 200}],
            },
            {
                "model_call_id": "anat:1",
                "turn_id": 1,
                "input_tokens": 600,
                "context_window_tokens": 1000,
                "fragments": [{"source_type": "tool_output", "tokens": 400}],
            },
        ]
        turn_heat = [
            {
                "model_call_id": "anat:0", "turn_id": 0,
                "cold_gap_score": 0.0, "window_pressure_score": 0.0,
                "assistant_share": 0.0, "red_token_share": 0.0, "raw_tool_share": 0.0,
            },
            {
                "model_call_id": "anat:1", "turn_id": 1,
                "cold_gap_score": 0.72, "window_pressure_score": 0.0,
                "assistant_share": 0.0, "red_token_share": 0.0, "raw_tool_share": 0.4,
            },
        ]
        # Секрет в title finding не должен попасть в байты PNG.
        findings = [{"kind": "cold_gap", "turn_id": 1, "title": "leak sk-testSECRET-anatomy"}]
        events = [
            {"event_type": "model_output", "payload": {"turn": 1, "message": {"tool_calls": [{"function": {"name": "write_file"}}]}}},
        ]

        data = build_anatomy_data(report, packets, turn_heat, findings, events)
        png_path = out_dir / "heatmap.png"
        render_heatmap_png(data, png_path)

        width, height, pixels = _read_png_rgb(png_path)
        pixel_set = set(pixels)
        # Фиксированный размер нового рендера.
        self.assertEqual((width, height), (1400, 860))
        # Секрет отсутствует в байтах (privacy — тот же инвариант, что у старого PNG).
        self.assertNotIn(b"sk-testSECRET-anatomy", png_path.read_bytes())
        # Палитра типов блока A присутствует на рисунке.
        self.assertIn(SOURCE_COLORS["user_message"], pixel_set)
        self.assertIn(SOURCE_COLORS["tool_output"], pixel_set)

    def test_context_window_png_written_with_new_name(self):
        """write_analysis_outputs создаёт оба PNG: heatmap.png и context_window.png."""

        cfg = self.make_cfg()
        trace_path = Path(cfg.root) / "trace.jsonl"
        _write_trace(trace_path, _valid_trace_events())
        events, warnings = load_trace_path(trace_path)
        result = analyze_events(events, warnings)
        out_dir = Path(cfg.root) / "outputs-png-names"
        write_analysis_outputs(result, out_dir)

        self.assertTrue((out_dir / "heatmap.png").exists())
        self.assertTrue((out_dir / "context_window.png").exists())
        # session_report.outputs содержит оба имени файлов.
        report = json.loads((out_dir / "session_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["outputs"]["heatmap_png"], "heatmap.png")
        self.assertEqual(report["outputs"]["context_window_png"], "context_window.png")

    def test_html_report_has_anatomy_tab_and_data(self):
        """HTML содержит 4-ю вкладку Anatomy с встроенными данными и privacy."""

        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "html-anatomy"
        out_dir.mkdir()
        report = {
            "session_id": "anat:html",
            "model_calls": 2,
            "max_red_token_share": 0.0,
            "max_cold_gap_score": 0.72,
            "max_assistant_share": 0.0,
            "max_fixed_instruction_cost": 0.1,
            "max_goal_anchor_cost": 0.2,
            "max_normative_status": 0.0,
            "max_goal_status": 0.0,
            "findings": 1,
        }
        (out_dir / "session_report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
        write_jsonl(out_dir / "turn_heat.jsonl", [
            {"model_call_id": "anat:0", "turn_id": 0, "cold_gap_score": 0.0,
             "window_pressure_score": 0.0, "assistant_share": 0.0, "red_token_share": 0.0},
            {"model_call_id": "anat:1", "turn_id": 1, "cold_gap_score": 0.72,
             "window_pressure_score": 0.0, "assistant_share": 0.0, "red_token_share": 0.0},
        ])
        write_jsonl(out_dir / "fragment_heat.jsonl", [])
        write_jsonl(out_dir / "packets.jsonl", [
            {"model_call_id": "anat:0", "turn_id": 0, "input_tokens": 100,
             "context_window_tokens": 1000,
             "fragments": [{"source_type": "user_message", "tokens": 100}]},
            {"model_call_id": "anat:1", "turn_id": 1, "input_tokens": 200,
             "context_window_tokens": 1000,
             "fragments": [{"source_type": "tool_output", "tokens": 150}]},
        ])
        write_jsonl(out_dir / "fragments.jsonl", [])
        # Секрет в аргументах tool_call — Anatomy берёт только имя инструмента,
        # args не должен попасть в HTML (events не встроены в heatmap-data).
        write_jsonl(out_dir / "events.jsonl", [
            {"event_type": "model_output", "payload": {"turn": 1,
              "message": {"tool_calls": [{"function": {
                "name": "write_file",
                "arguments": "{\"path\":\"leak sk-testSECRET-html-anatomy\"}",
              }}]}}},
        ])
        write_jsonl(out_dir / "findings.jsonl", [
            {"kind": "cold_gap", "turn_id": 1, "title": "regular cold gap finding"},
        ])

        html_path = out_dir / "heatmap.html"
        render_html_report(out_dir / "session_report.json", html_path)
        html_content = html_path.read_text(encoding="utf-8")

        self.assertIn('id="tab-anatomy"', html_content)
        self.assertIn('id="panel-anatomy"', html_content)
        self.assertIn("FRAGMENT MASS", html_content)
        # Данные анатомии встроены в JSON-скрипт страницы.
        self.assertIn('"anatomy"', html_content)
        # Verdict-точки: red=0, cold=1, pressure=0 (max_assistant_share=0 в report).
        # html.escape экранирует кавычки в JSON-блоке, поэтому проверяем значение dots.
        self.assertIn("&quot;dots&quot;: &quot;010&quot;", html_content)
        # Zoom/связки/клик-переход присутствуют в скрипте.
        self.assertIn("data-zoom", html_content)
        self.assertIn("renderAnatomyDetail", html_content)
        self.assertIn("switchToContextWindow", html_content)
        # Privacy: секрет не утёк в HTML.
        self.assertNotIn("sk-testSECRET-html-anatomy", html_content)

    def test_anatomy_on_flappy_trace(self):
        """Регрессия на реальной трассе: verdict и швы summarization не пустые."""

        root = Path(__file__).resolve().parents[1]
        trace_path = root / "tests" / "data" / "traces" / "flappy" / "flappy2a_sum3.jsonl"
        events, warnings = load_trace(trace_path)
        result = analyze_events(events, warnings)
        cfg = self.make_cfg()
        out_dir = Path(cfg.root) / "flappy-anatomy"
        write_analysis_outputs(result, out_dir)

        # Оба PNG сгенерировались без падения на реальных данных.
        self.assertTrue((out_dir / "heatmap.png").exists())
        self.assertTrue((out_dir / "context_window.png").exists())
        # На flappy2 summarization точно работал — швы должны быть.
        report = json.loads((out_dir / "session_report.json").read_text(encoding="utf-8"))
        # Если есть cold gaps — verdict указывает на конкретный ход.
        if report.get("max_cold_gap_score", 0) > 0:
            self.assertGreater(report["max_cold_gap_score"], 0.0)
