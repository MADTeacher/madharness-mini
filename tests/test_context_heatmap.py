import json
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from context_heatmap.cli import main as heatmap_main
from context_heatmap.features import detect_cold_gaps, score_fragment
from context_heatmap.io import read_jsonl
from context_heatmap.loaders.madharness_trace import load_trace
from context_heatmap.normalize import load_normalized_events
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
        for filename in report["outputs"].values():
            self.assertTrue((out_dir / filename).exists(), filename)
        self.assertEqual(
            (out_dir / "fragment_heat.csv").read_text(encoding="utf-8").splitlines()[0],
            "session_id,model_call_id,fragment_id,heat,confidence,axes,reasons",
        )
        self.assertEqual(
            (out_dir / "turn_heat.csv").read_text(encoding="utf-8").splitlines()[0],
            "session_id,model_call_id,turn_id,red_token_share,stale_token_share,"
            "raw_tool_share,evidence_density,cold_gap_score,taint_exposure,top_reasons",
        )
        self.assertIn("heatmap-data", html)
        self.assertIn("cold", html)
        self.assertIn("reasons=", html)
        self.assertIn("Context Heatmap Report", markdown)
        self.assertIn("<redacted>", fragments)
        self.assertNotIn("sk-testSECRET123456789", fragments)
        self.assertNotIn("sk-testSECRET123456789", html)
        self.assertNotIn("sk-testSECRET123456789", markdown)
        self.assertNotIn("sk-testSECRET123456789", json.dumps(report))
        self.assertTrue(any(finding.kind == "cold_gap" for finding in result.findings))

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
        self.assertTrue(rendered.exists())
        self.assertTrue((batch_dir / "valid" / "session_report.json").exists())
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
