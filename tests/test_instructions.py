import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from madharness_mini.config import Config
from madharness_mini.instructions import (
    PROJECT_DOC_MAX_BYTES,
    load_project_instructions,
    load_prompt,
)
from madharness_mini.loop import base_messages

from tests.helpers import HarnessTestCase


class InstructionTests(HarnessTestCase):
    def test_base_messages_loads_system_prompt_from_markdown(self):
        cfg = self.make_cfg()
        messages = base_messages(cfg, "Return a short greeting")
        system_prompt = load_prompt("system")

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertIn("You are madharness-mini", system_prompt)
        self.assertIn("# Tool use", system_prompt)
        self.assertIn("Never invent tool results", system_prompt)
        self.assertEqual(
            messages[1], {"role": "user", "content": "Return a short greeting"}
        )

    def test_base_messages_appends_root_agents_md(self):
        cfg = self.make_cfg()
        (cfg.root / "AGENTS.md").write_text(
            "Use the project test command.\n", encoding="utf-8"
        )

        messages = base_messages(cfg, "Return a short greeting")

        self.assertTrue(messages[0]["content"].startswith(load_prompt("system")))
        self.assertIn("# Project instructions", messages[0]["content"])
        self.assertIn("Use the project test command.", messages[0]["content"])

    def test_empty_agents_md_is_ignored(self):
        cfg = self.make_cfg()
        (cfg.root / "AGENTS.md").write_text("  \n\n", encoding="utf-8")

        self.assertEqual(load_project_instructions(cfg), "")
        self.assertEqual(base_messages(cfg, "hello")[0]["content"], load_prompt("system"))

    def test_nested_agents_md_is_appended_after_root_file(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        active = root / "services" / "payments"
        active.mkdir(parents=True)
        (active / ".madharness-mini").mkdir()
        (active / ".madharness-mini" / "config.json").write_text(
            json.dumps({"workspace_root": "../..", "allow_shell": True}),
            encoding="utf-8",
        )
        (root / "AGENTS.md").write_text("Root rules", encoding="utf-8")
        (root / "services" / "AGENTS.md").write_text("Service rules", encoding="utf-8")
        (active / "AGENTS.md").write_text("Payment rules", encoding="utf-8")
        cfg = Config(active)
        self.addCleanup(tmp.cleanup)

        instructions = load_project_instructions(cfg)

        self.assertEqual(instructions, "Root rules\n\nService rules\n\nPayment rules")

    def test_agents_override_is_ignored(self):
        cfg = self.make_cfg()
        (cfg.root / "AGENTS.override.md").write_text("Override rules", encoding="utf-8")
        (cfg.root / "AGENTS.md").write_text("Normal rules", encoding="utf-8")

        self.assertEqual(load_project_instructions(cfg), "Normal rules")

    def test_global_agents_md_is_ignored(self):
        cfg = self.make_cfg()
        home = tempfile.TemporaryDirectory()
        home_path = Path(home.name)
        (home_path / ".codex").mkdir()
        (home_path / ".codex" / "AGENTS.md").write_text(
            "Global rules", encoding="utf-8"
        )
        self.addCleanup(home.cleanup)

        with patch.dict(os.environ, {"HOME": str(home_path), "CODEX_HOME": str(home_path)}):
            self.assertEqual(load_project_instructions(cfg), "")

    def test_agents_md_is_limited_to_codex_default_size(self):
        cfg = self.make_cfg()
        (cfg.root / "AGENTS.md").write_text(
            "a" * (PROJECT_DOC_MAX_BYTES + 10), encoding="utf-8"
        )

        instructions = load_project_instructions(cfg)

        self.assertEqual(len(instructions.encode("utf-8")), PROJECT_DOC_MAX_BYTES)
        self.assertEqual(instructions, "a" * PROJECT_DOC_MAX_BYTES)
