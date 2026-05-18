from madharness_mini.tools import ToolRegistry

from tests.helpers import HarnessTestCase


class ToolTests(HarnessTestCase):
    def test_read_file_tool(self):
        cfg = self.make_cfg()
        (cfg.root / "hello.txt").write_text("one\ntwo\n", encoding="utf-8")
        obs = ToolRegistry(cfg).call(
            "read_file", {"path": "hello.txt", "start": 1, "end": 1}
        )
        self.assertTrue(obs["ok"])
        self.assertIn("1: one", obs["content"])

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

    def test_replace_text_tool_replaces_exact_text(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "replace_text",
            {"path": "hello.txt", "old": "two", "new": "deux"},
        )

        self.assertTrue(obs["ok"])
        self.assertEqual(obs["replacements"], 1)
        self.assertEqual(path.read_text(encoding="utf-8"), "one\ndeux\n")

    def test_replace_text_tool_fails_when_text_is_missing(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "replace_text",
            {"path": "hello.txt", "old": "three", "new": "trois"},
        )

        self.assertFalse(obs["ok"])
        self.assertIn("expected 1 replacements, found 0", obs["summary"])
        self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")

    def test_replace_text_tool_fails_on_unexpected_duplicate_match(self):
        cfg = self.make_cfg()
        path = cfg.root / "hello.txt"
        path.write_text("same\nsame\n", encoding="utf-8")

        obs = ToolRegistry(cfg).call(
            "replace_text",
            {"path": "hello.txt", "old": "same", "new": "changed"},
        )

        self.assertFalse(obs["ok"])
        self.assertIn("expected 1 replacements, found 2", obs["summary"])
        self.assertEqual(path.read_text(encoding="utf-8"), "same\nsame\n")

    def test_replace_text_tool_respects_path_policy(self):
        obs = ToolRegistry(self.make_cfg()).call(
            "replace_text", {"path": "../nope.txt", "old": "a", "new": "b"}
        )
        self.assertFalse(obs["ok"])

    def test_apply_patch_is_registered(self):
        schemas = ToolRegistry(self.make_cfg()).schemas()
        names = [item["function"]["name"] for item in schemas]
        self.assertIn("apply_patch", names)
        self.assertIn("replace_text", names)

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
        self.assertEqual((cfg.root / "added.txt").read_text(encoding="utf-8"), "hello\nworld\n")

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
        self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")

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
