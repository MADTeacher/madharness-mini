from pathlib import Path

from madharness_mini.tools import ToolRegistry
from madharness_mini.tools.specs import ToolSpec
from madharness_mini.utils import obj

from tests.helpers import HarnessTestCase

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


class ToolTests(HarnessTestCase):
    def tool_schema_text(self, name):
        schemas = ToolRegistry(self.make_cfg()).schemas()
        schema = next(
            item["function"] for item in schemas if item["function"]["name"] == name
        )
        parts = [schema["description"]]
        for prop in schema["parameters"]["properties"].values():
            parts.append(prop.get("description", ""))
        return "\n".join(parts)

    def test_read_file_tool(self):
        cfg = self.make_cfg()
        (cfg.root / "hello.txt").write_text("one\ntwo\n", encoding="utf-8")
        obs = ToolRegistry(cfg).call(
            "read_file", {"path": "hello.txt", "start": 1, "end": 1}
        )
        self.assertTrue(obs["ok"])
        self.assertIn("1: one", obs["content"])

    def test_read_image_tool_returns_metadata_without_base64(self):
        cfg = self.make_cfg()
        (cfg.root / "shot.png").write_bytes(PNG_BYTES)

        obs = ToolRegistry(cfg).call("read_image", {"path": "shot.png"})

        self.assertTrue(obs["ok"])
        self.assertEqual(obs["mime_type"], "image/png")
        self.assertEqual(obs["bytes"], len(PNG_BYTES))
        self.assertEqual(obs["detail"], "auto")
        self.assertFalse(obs["attached"])
        self.assertNotIn("data:image", str(obs))

    def test_read_image_tool_marks_attachment_when_enabled(self):
        cfg = self.make_cfg()
        cfg.data["supports_image_input"] = True
        (cfg.root / "shot.png").write_bytes(PNG_BYTES)

        obs = ToolRegistry(cfg).call(
            "read_image", {"path": "shot.png", "detail": "high"}
        )

        self.assertTrue(obs["ok"])
        self.assertTrue(obs["attached"])
        self.assertEqual(obs["detail"], "high")

    def test_read_image_tool_rejects_policy_and_bad_inputs(self):
        cfg = self.make_cfg()
        (cfg.root / ".env").write_bytes(PNG_BYTES)
        (cfg.root / "bad.txt").write_bytes(PNG_BYTES)

        protected = ToolRegistry(cfg).call("read_image", {"path": ".env"})
        outside = ToolRegistry(cfg).call("read_image", {"path": "../shot.png"})
        unsupported = ToolRegistry(cfg).call("read_image", {"path": "bad.txt"})
        invalid_detail = ToolRegistry(cfg).call(
            "read_image", {"path": "bad.txt", "detail": "microscope"}
        )

        self.assertFalse(protected["ok"])
        self.assertFalse(outside["ok"])
        self.assertFalse(unsupported["ok"])
        self.assertFalse(invalid_detail["ok"])

    def test_read_image_tool_rejects_large_file(self):
        cfg = self.make_cfg()
        cfg.data["max_image_bytes"] = 4
        (cfg.root / "shot.png").write_bytes(PNG_BYTES)

        obs = ToolRegistry(cfg).call("read_image", {"path": "shot.png"})

        self.assertFalse(obs["ok"])
        self.assertIn("too large", obs["summary"])

    def test_write_file_tool_creates_parent_dirs(self):
        cfg = self.make_cfg()
        obs = ToolRegistry(cfg).call(
            "write_file", {"path": "example/hello.txt", "content": "hello\n"}
        )
        self.assertTrue(obs["ok"])
        self.assertEqual(
            (cfg.root / "example" / "hello.txt").read_text(encoding="utf-8"), "hello\n"
        )

    def test_write_file_tool_respects_path_policy(self):
        obs = ToolRegistry(self.make_cfg()).call(
            "write_file", {"path": "../nope.txt", "content": "bad"}
        )
        self.assertFalse(obs["ok"])

    def test_apply_patch_is_registered(self):
        schemas = ToolRegistry(self.make_cfg()).schemas()
        names = [item["function"]["name"] for item in schemas]
        self.assertEqual(
            names,
            [
                "list_files",
                "read_file",
                "read_image",
                "write_file",
                "apply_patch",
                "search_code",
                "run_shell",
            ],
        )

    def test_registry_accepts_extra_tool_provider(self):
        class ExtraProvider:
            def specs(self, ctx):
                return [
                    ToolSpec(
                        "extra_tool",
                        "Extra tool for registry tests.",
                        obj({}),
                        lambda ctx, args: {
                            "ok": True,
                            "tool": "extra_tool",
                            "summary": "extra worked",
                        },
                    )
                ]

        registry = ToolRegistry(self.make_cfg(), providers=[ExtraProvider()])
        names = [item["function"]["name"] for item in registry.schemas()]

        self.assertIn("extra_tool", names)
        self.assertEqual(
            registry.call("extra_tool", {}),
            {"ok": True, "tool": "extra_tool", "summary": "extra worked"},
        )

    def test_registry_rejects_duplicate_tool_names(self):
        class DuplicateProvider:
            def specs(self, ctx):
                return [
                    ToolSpec(
                        "read_file",
                        "Duplicate tool.",
                        obj({}),
                        lambda ctx, args: {
                            "ok": True,
                            "tool": "read_file",
                            "summary": "duplicate",
                        },
                    )
                ]

        with self.assertRaisesRegex(RuntimeError, "duplicate tool name: read_file"):
            ToolRegistry(self.make_cfg(), providers=[DuplicateProvider()])

    def test_apply_patch_schema_describes_strict_patch_format(self):
        combined = self.tool_schema_text("apply_patch")

        self.assertIn("*** Begin Patch", combined)
        self.assertIn("*** End Patch", combined)
        self.assertIn("*** Update File:", combined)
        self.assertIn("*** Add File:", combined)
        self.assertIn("*** Delete File:", combined)
        self.assertIn("*** Move to:", combined)
        self.assertIn("multiline string", combined)
        self.assertIn("not a shell command", combined)
        self.assertIn("context line begins with one space", combined)
        self.assertIn("-removed line begins with minus", combined)
        self.assertIn("+added line begins with plus", combined)
        self.assertIn("reread the current file region", combined)
        self.assertIn("retry apply_patch", combined)

    def test_list_files_schema_describes_scope_and_limits(self):
        combined = self.tool_schema_text("list_files")

        self.assertIn("Recursively list files", combined)
        self.assertIn("files only", combined)
        self.assertIn("ignored folders", combined)
        self.assertIn("200", combined)
        self.assertIn("file name", combined)
        self.assertIn("defaults to .", combined)

    def test_read_file_schema_describes_numbered_excerpts(self):
        combined = self.tool_schema_text("read_file")

        self.assertIn("UTF-8 file excerpt", combined)
        self.assertIn("1-based line numbers", combined)
        self.assertIn("numbered lines", combined)
        self.assertIn("Workspace-relative file path", combined)

    def test_read_image_schema_describes_text_only_fallback(self):
        combined = self.tool_schema_text("read_image")

        self.assertIn("PNG, JPEG, WEBP", combined)
        self.assertIn("non-animated GIF", combined)
        self.assertIn("metadata only", combined)
        self.assertIn("supports_image_input", combined)
        self.assertIn("must not claim", combined)

    def test_write_file_schema_warns_about_full_overwrite(self):
        combined = self.tool_schema_text("write_file")

        self.assertIn("complete UTF-8 text file", combined)
        self.assertIn("fully overwrites", combined)
        self.assertIn("creates parent directories", combined)
        self.assertIn("Prefer apply_patch", combined)
        self.assertIn("Do not use write_file as", combined)
        self.assertIn("failed precise edit", combined)

    def test_search_code_schema_describes_literal_search(self):
        combined = self.tool_schema_text("search_code")

        self.assertIn("literal substring", combined)
        self.assertIn("not regex", combined)
        self.assertIn("not semantic search", combined)
        self.assertIn("100 matches", combined)
        self.assertIn("file names only", combined)

    def test_run_shell_schema_describes_policy_limits(self):
        combined = self.tool_schema_text("run_shell")

        self.assertIn("one allowed command", combined)
        self.assertIn("workspace root", combined)
        self.assertIn("60 seconds", combined)
        self.assertIn("single command", combined)
        self.assertIn("shell control operators", combined)
        self.assertIn("rm -rf", combined)
        self.assertIn("Do not use run_shell to edit files", combined)

    def test_system_prompt_describes_tool_recovery_rules(self):
        prompt = Path("madharness_mini/prompts/system.md").read_text(encoding="utf-8")

        self.assertIn("If `apply_patch` fails", prompt)
        self.assertIn("verbatim context", prompt)
        self.assertIn("not for editing files", prompt)
        self.assertIn("`command` argument of `run_shell`", prompt)
        self.assertIn("never use a command itself as a tool name", prompt)
        self.assertIn("same tool returns the same error", prompt)

    def test_unknown_tool_returns_fail_observation(self):
        obs = ToolRegistry(self.make_cfg()).call("missing_tool", {})

        self.assertFalse(obs["ok"])
        self.assertEqual(obs["tool"], "missing_tool")
        self.assertEqual(obs["summary"], "unknown tool")

    def test_apply_patch_updates_one_line_with_context(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\ntwo\nthree\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "@@",
                        " one",
                        "-two",
                        "+deux",
                        " three",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertEqual(path.read_text(encoding="utf-8"), "one\ndeux\nthree\n")

    def test_apply_patch_updates_multiline_hunk(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("start\nalpha\nbeta\nend\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "@@",
                        " start",
                        "-alpha",
                        "-beta",
                        "+one",
                        "+two",
                        " end",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertEqual(path.read_text(encoding="utf-8"), "start\none\ntwo\nend\n")

    def test_apply_patch_adds_file(self):
        cfg = self.make_cfg()

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: added.txt",
                        "+hello",
                        "+world",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertEqual(
            (cfg.root / "added.txt").read_text(encoding="utf-8"), "hello\nworld\n"
        )

    def test_apply_patch_deletes_file(self):
        cfg = self.make_cfg()
        path = cfg.root / "delete-me.txt"
        path.write_text("bye\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Delete File: delete-me.txt",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertFalse(path.exists())

    def test_apply_patch_fails_on_ambiguous_context_without_writing(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("same\nold\nsame\nold\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "@@",
                        " same",
                        "-old",
                        "+new",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("expected 1 hunk match, found 2", obs["summary"])
        self.assertTrue(obs["retryable"])
        self.assertIn("Add more surrounding context", obs["hint"])
        self.assertEqual(path.read_text(encoding="utf-8"), "same\nold\nsame\nold\n")

    def test_apply_patch_fails_on_missing_context_without_writing(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "@@",
                        "-missing",
                        "+found",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("expected 1 hunk match, found 0", obs["summary"])
        self.assertTrue(obs["retryable"])
        self.assertIn("reread the exact region", obs["hint"])
        self.assertIn("verbatim current context", obs["hint"])
        self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")

    def test_apply_patch_format_failure_explains_patch_boundaries(self):
        obs = ToolRegistry(self.make_cfg()).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "apply_patch <<'PATCH'",
                        "*** Begin Patch",
                        "*** End Patch",
                        "PATCH",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertEqual(obs["summary"], "patch must start with *** Begin Patch")
        self.assertTrue(obs["retryable"])
        self.assertIn("starting with *** Begin Patch", obs["hint"])
        self.assertIn("ending with *** End Patch", obs["hint"])
        self.assertIn("Do not wrap it in a shell command", obs["hint"])

    def test_apply_patch_blank_hunk_line_explains_context_marker(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\n\ntwo\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "@@",
                        " one",
                        "",
                        "-two",
                        "+deux",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertEqual(obs["summary"], "invalid hunk line: ")
        self.assertTrue(obs["retryable"])
        self.assertIn("Blank context lines", obs["hint"])
        self.assertIn("one leading space", obs["hint"])
        self.assertEqual(path.read_text(encoding="utf-8"), "one\n\ntwo\n")

    def test_apply_patch_moves_file_without_hunk(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\n", encoding="utf-8")
        target = cfg.root / "renamed.txt"

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "*** Move to: renamed.txt",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertFalse(path.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "one\n")

    def test_apply_patch_moves_file_and_applies_hunk(self):
        cfg = self.make_cfg()
        path = cfg.root / "old" / "hello.txt"
        path.parent.mkdir()
        path.write_text("one\ntwo\n", encoding="utf-8")
        target = cfg.root / "new" / "hello.txt"

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: old/hello.txt",
                        "*** Move to: new/hello.txt",
                        "@@",
                        "-one",
                        "+uno",
                        " two",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertFalse(path.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "uno\ntwo\n")

    def test_apply_patch_move_fails_when_target_exists_without_writing(self):
        cfg = self.make_cfg()
        source = cfg.root / "source.txt"
        target = cfg.root / "target.txt"
        source.write_text("source\n", encoding="utf-8")
        target.write_text("target\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: source.txt",
                        "*** Move to: target.txt",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("target file already exists: target.txt", obs["summary"])
        self.assertEqual(source.read_text(encoding="utf-8"), "source\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "target\n")

    def test_apply_patch_move_fails_when_target_is_outside_workspace(self):
        cfg = self.make_cfg()
        source = cfg.root / "source.txt"
        source.write_text("source\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: source.txt",
                        "*** Move to: ../target.txt",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("outside workspace", obs["summary"])
        self.assertEqual(source.read_text(encoding="utf-8"), "source\n")

    def test_apply_patch_move_fails_when_target_is_protected(self):
        cfg = self.make_cfg()
        source = cfg.root / "source.txt"
        source.write_text("source\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: source.txt",
                        "*** Move to: .env",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("protected path", obs["summary"])
        self.assertEqual(source.read_text(encoding="utf-8"), "source\n")

    def test_apply_patch_move_fails_when_source_is_missing(self):
        cfg = self.make_cfg()

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: missing.txt",
                        "*** Move to: target.txt",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("not a file: missing.txt", obs["summary"])
        self.assertFalse((cfg.root / "target.txt").exists())

    def test_apply_patch_move_to_fails_outside_update_file(self):
        obs = ToolRegistry(self.make_cfg()).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Move to: target.txt",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("Move to is only supported after Update File", obs["summary"])

    def test_apply_patch_accepts_optional_end_of_file_marker(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: hello.txt",
                        "@@",
                        "-one",
                        "+uno",
                        "*** End of File",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertTrue(obs["ok"])
        self.assertEqual(path.read_text(encoding="utf-8"), "uno\ntwo\n")

    def test_apply_patch_respects_path_policy(self):
        obs = ToolRegistry(self.make_cfg()).call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: ../nope.txt",
                        "+bad",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(obs["ok"])
        self.assertIn("outside workspace", obs["summary"])
