import json
import struct
import zlib
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from context_heatmap.cli import main as heatmap_main
from context_heatmap.features import detect_cold_gaps, score_fragment
from context_heatmap.io import read_jsonl
from context_heatmap.loaders.madharness_trace import load_trace
from context_heatmap.normalize import load_normalized_events
from context_heatmap.png import SOURCE_COLORS, render_png_summary
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
            "raw_tool_share,evidence_density,cold_gap_score,taint_exposure,"
            "fixed_instruction_cost,goal_anchor_cost,normative_status,goal_status,"
            "instruction_scope_score,goal_supersession_score,"
            "attached_data_taint_score,top_reasons",
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
        render_png_summary(report, packets, turn_heat, findings, png_path)
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

    def test_docs_link_context_heatmap_roadmap(self):
        root = Path(__file__).resolve().parents[1]
        docs_index = (root / "docs" / "README.md").read_text(encoding="utf-8")
        capabilities = (root / "docs" / "capabilities.md").read_text(encoding="utf-8")
        context_layer = (root / "docs" / "context-layer.md").read_text(encoding="utf-8")

        self.assertIn("context-heatmap-roadmap.md", docs_index)
        self.assertIn("пассивной диагностики", capabilities)
        self.assertIn("context_packet", context_layer)
