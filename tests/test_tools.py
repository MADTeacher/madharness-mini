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

    def test_write_scope_limits_write_file_to_allowed_suffixes(self):
        cfg = self.make_cfg()
        registry = ToolRegistry(
            cfg,
            writable_suffixes=(".md",),
            write_scope_description="planner may write only Markdown plan files (.md)",
        )

        denied = registry.call(
            "write_file", {"path": "index.html", "content": "<html></html>\n"}
        )
        allowed = registry.call(
            "write_file", {"path": "PLAN.md", "content": "# Plan\n"}
        )

        self.assertFalse(denied["ok"])
        self.assertIn("planner may write only Markdown plan files", denied["summary"])
        self.assertFalse((cfg.root / "index.html").exists())
        self.assertTrue(allowed["ok"])
        self.assertEqual((cfg.root / "PLAN.md").read_text(encoding="utf-8"), "# Plan\n")

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
                "run_shell_background",
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
        # По умолчанию команды идут через /bin/sh -c, операторы работают.
        self.assertIn("/bin/sh -c", combined)
        self.assertIn("&&", combined)
        self.assertIn("2>&1", combined)
        # Чёрный список разрушительных команд описан.
        self.assertIn("rm -rf", combined)
        self.assertIn("sudo", combined)
        # Редирект в файлы запрещён — модель должна знать, что для правки файлов
        # есть apply_patch/write_file, а не shell-перенаправление.
        self.assertIn("apply_patch", combined)
        self.assertIn("write_file", combined)
        # Подсказка про allow_shell_interpreter и фон через run_shell_background.
        self.assertIn("allow_shell_interpreter", combined)
        self.assertIn("run_shell_background", combined)

    def test_run_shell_accepts_workspace_relative_cwd(self):
        cfg = self.make_cfg()
        (cfg.root / "sub").mkdir()

        obs = ToolRegistry(cfg).call("run_shell", {"command": "pwd", "cwd": "sub"})

        self.assertTrue(obs["ok"])
        self.assertEqual(obs["cwd"], "sub")
        self.assertEqual(obs["returncode"], 0)

    def test_run_shell_saves_full_output_to_file_on_truncation(self):
        """При обрезке длинного stdout полный вывод уходит в файл, модели — хвост."""
        import subprocess
        from unittest.mock import patch

        from madharness_mini.utils import MAX_OUTPUT

        cfg = self.make_cfg()
        cfg.ensure_dirs()
        huge_stdout = "BEGIN_HEAD_LINE\n" + "x" * (MAX_OUTPUT + 5000)
        fake_proc = subprocess.CompletedProcess(
            args=["echo"], returncode=0, stdout=huge_stdout, stderr=""
        )
        with patch(
            "madharness_mini.tools.shell.subprocess.run", return_value=fake_proc
        ):
            obs = ToolRegistry(cfg).call("run_shell", {"command": "echo big"})

        self.assertTrue(obs["ok"])
        self.assertTrue(obs["stdout_truncated"])
        self.assertEqual(obs["stdout_original_chars"], len(huge_stdout))
        # Модели отдали хвост, а не голову.
        self.assertIn("x", obs["stdout"])
        self.assertNotIn("BEGIN_HEAD_LINE", obs["stdout"])
        # Полный вывод сохранён, файл содержит начало (голову).
        full_path = Path(obs["stdout_full_path"])
        self.assertTrue(full_path.is_file())
        self.assertEqual(full_path.read_text(encoding="utf-8"), huge_stdout)

    def test_run_shell_no_truncation_when_output_below_limit(self):
        """Короткий вывод не помечается обрезанным, файл не создаётся."""
        import subprocess
        from unittest.mock import patch

        cfg = self.make_cfg()
        cfg.ensure_dirs()
        fake_proc = subprocess.CompletedProcess(
            args=["echo"], returncode=0, stdout="short output\n", stderr=""
        )
        with patch(
            "madharness_mini.tools.shell.subprocess.run", return_value=fake_proc
        ):
            obs = ToolRegistry(cfg).call("run_shell", {"command": "echo short"})

        self.assertTrue(obs["ok"])
        self.assertNotIn("stdout_truncated", obs)
        self.assertNotIn("stdout_full_path", obs)
        self.assertEqual(obs["stdout"], "short output\n")

    def test_run_shell_respects_disabled_full_output_flag(self):
        """При context_shell_full_output=False — прежняя семантика головы, файла нет."""
        import subprocess
        from unittest.mock import patch

        from madharness_mini.utils import MAX_OUTPUT

        cfg = self.make_cfg()
        cfg.data["context_shell_full_output"] = False
        cfg.ensure_dirs()
        huge_stdout = "HEAD_LINE\n" + "y" * (MAX_OUTPUT + 1000)
        fake_proc = subprocess.CompletedProcess(
            args=["echo"], returncode=0, stdout=huge_stdout, stderr=""
        )
        with patch(
            "madharness_mini.tools.shell.subprocess.run", return_value=fake_proc
        ):
            obs = ToolRegistry(cfg).call("run_shell", {"command": "echo big"})

        self.assertTrue(obs["ok"])
        # Прежняя семантика: голова через clipped, без признаков обрезки и файла.
        self.assertIn("HEAD_LINE", obs["stdout"])
        self.assertIn("[clipped", obs["stdout"])
        self.assertNotIn("stdout_truncated", obs)
        self.assertNotIn("stdout_full_path", obs)
        # Файл не должен был создаться.
        outputs_dir = cfg.state_dir / "tool_outputs"
        log_files = list(outputs_dir.glob("shell-*.log"))
        self.assertEqual(log_files, [])

    def test_system_prompt_describes_tool_recovery_rules(self):
        prompt = Path("madharness_mini/prompts/system.md").read_text(encoding="utf-8")

        self.assertIn("If `apply_patch` fails", prompt)
        self.assertIn("verbatim context", prompt)
        self.assertIn("not for editing files", prompt)
        self.assertIn("`command` argument of `run_shell`", prompt)
        self.assertIn("never use a command itself as a tool name", prompt)
        self.assertIn("same tool returns the same error", prompt)
        # Новый режим по умолчанию: /bin/sh -c, операторы работают; фон через
        # run_shell_background упомянут. Brace expansion по-прежнему блокируется.
        self.assertIn("/bin/sh -c", prompt)
        self.assertIn("Brace expansion", prompt)
        self.assertIn("run_shell_background", prompt)

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

    def test_write_scope_limits_apply_patch_paths(self):
        cfg = self.make_cfg()
        registry = ToolRegistry(
            cfg,
            writable_suffixes=(".md",),
            write_scope_description="planner may write only Markdown plan files (.md)",
        )

        denied = registry.call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: index.html",
                        "+<html></html>",
                        "*** End Patch",
                    ]
                )
            },
        )
        allowed = registry.call(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: PLAN.md",
                        "+# Plan",
                        "*** End Patch",
                    ]
                )
            },
        )

        self.assertFalse(denied["ok"])
        self.assertIn("planner may write only Markdown plan files", denied["summary"])
        self.assertFalse((cfg.root / "index.html").exists())
        self.assertTrue(allowed["ok"])
        self.assertEqual((cfg.root / "PLAN.md").read_text(encoding="utf-8"), "# Plan\n")

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


class RunShellBackgroundTests(HarnessTestCase):
    """Lifecycle фонового shell-процесса: start/status/stop + auto-cleanup."""

    def test_start_status_stop_roundtrip(self):
        cfg = self.make_cfg()
        registry = ToolRegistry(cfg)
        try:
            # Команда, которая пишет в stdout и сама завершается за <1с.
            start = registry.call(
                "run_shell_background",
                {"action": "start", "command": "echo hello && sleep 1"},
            )
            self.assertTrue(start["ok"])
            bg_id = start["id"]
            self.assertIn("pid", start)

            status = registry.call(
                "run_shell_background", {"action": "status", "id": bg_id}
            )
            self.assertTrue(status["ok"])
            # running True/False зависит от тайминга; важен формат ответа.
            self.assertIn("running", status)

            stop = registry.call(
                "run_shell_background", {"action": "stop", "id": bg_id}
            )
            self.assertTrue(stop["ok"])
            self.assertIn("exit_code", stop)
        finally:
            registry.close()

    def test_status_unknown_id_fails(self):
        registry = ToolRegistry(self.make_cfg())
        try:
            obs = registry.call(
                "run_shell_background", {"action": "status", "id": "ghost"}
            )
            self.assertFalse(obs["ok"])
            self.assertIn("ghost", obs["summary"])
        finally:
            registry.close()

    def test_start_denied_by_policy(self):
        registry = ToolRegistry(self.make_cfg())
        try:
            obs = registry.call(
                "run_shell_background", {"action": "start", "command": "rm -rf /"}
            )
            self.assertFalse(obs["ok"])
            self.assertIn("risky", obs["summary"])
        finally:
            registry.close()

    def test_start_denies_redirect_into_file(self):
        # Редирект в файл — обход apply_patch/write_file, блокируется той же
        # политикой, что и у run_shell.
        registry = ToolRegistry(self.make_cfg())
        try:
            obs = registry.call(
                "run_shell_background",
                {"action": "start", "command": "echo x > evil.txt"},
            )
            self.assertFalse(obs["ok"])
            self.assertIn("redirect", obs["summary"])
        finally:
            registry.close()

    def test_max_background_processes_limit(self):
        cfg = self.make_cfg()
        cfg.data["max_background_processes"] = 2
        registry = ToolRegistry(cfg)
        try:
            self.assertTrue(
                registry.call(
                    "run_shell_background",
                    {"action": "start", "command": "sleep 30"},
                )["ok"]
            )
            self.assertTrue(
                registry.call(
                    "run_shell_background",
                    {"action": "start", "command": "sleep 30"},
                )["ok"]
            )
            # Третий превышает лимит — отказ.
            third = registry.call(
                "run_shell_background",
                {"action": "start", "command": "sleep 30"},
            )
            self.assertFalse(third["ok"])
            self.assertIn("limit", third["summary"])
        finally:
            registry.close()

    def test_registry_close_stops_background_processes(self):
        # Главный инвариант: cleanup при session_end гасит висящие процессы.
        cfg = self.make_cfg()
        registry = ToolRegistry(cfg)
        start = registry.call(
            "run_shell_background", {"action": "start", "command": "sleep 60"}
        )
        self.assertTrue(start["ok"])
        pid = start["pid"]

        registry.close()

        import os
        import time

        time.sleep(0.3)
        # kill -0 проверяет существование процесса без отправки сигнала.
        alive = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        self.assertFalse(alive)
