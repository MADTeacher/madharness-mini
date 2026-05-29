import json
from pathlib import Path
from unittest.mock import patch

from madharness_mini.loop import ask, run_agent
from madharness_mini.skills import (
    SkillRuntime,
    discover_skills,
    find_explicit_skill_selection,
)
from madharness_mini.skills.activation import list_skill_resources
from madharness_mini.cli import skills_command

from tests.helpers import HarnessTestCase


def write_skill(root: Path, name: str, description: str = "Пишет документацию.") -> None:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "license: MIT",
                "compatibility: Requires only local files",
                "metadata:",
                "  author: test",
                "allowed-tools: Read Bash(python:*)",
                "---",
                "",
                "# Docs Writer",
                "",
                "Пиши коротко и проверяемо.",
            ]
        ),
        encoding="utf-8",
    )


class SkillTests(HarnessTestCase):
    def test_discover_skills_reads_frontmatter_and_native_overrides_agents(self):
        cfg = self.make_cfg()
        write_skill(cfg.root / ".agents" / "skills" / "docs-writer", "docs-writer")
        write_skill(
            cfg.root / ".madharness_mini" / "skills" / "docs-writer",
            "docs-writer",
            "Нативная версия.",
        )

        index = discover_skills(cfg)

        self.assertEqual(index.names(), ["docs-writer"])
        skill = index.skills["docs-writer"]
        self.assertEqual(skill.source, "native")
        self.assertEqual(skill.description, "Нативная версия.")
        self.assertEqual(skill.metadata["author"], "test")
        self.assertEqual(skill.allowed_tools, ("Read", "Bash(python:*)"))
        self.assertTrue(
            any("shadowed" in item.message for item in index.diagnostics)
        )

    def test_discover_skips_invalid_skill_without_breaking_empty_projects(self):
        cfg = self.make_cfg()
        bad = cfg.root / ".agents" / "skills" / "broken"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text(
            "---\nname: broken\n---\nbody\n", encoding="utf-8"
        )

        index = discover_skills(cfg)

        self.assertEqual(index.skills, {})
        self.assertTrue(
            any("missing required description" in item.message for item in index.diagnostics)
        )

    def test_explicit_skill_selection_supports_markers_and_phrases(self):
        selection = find_explicit_skill_selection(
            "@skill:docs-writer @skill/docs-writer используй навык test-skill $missing-skill",
            {"docs-writer", "test-skill"},
        )

        self.assertEqual(selection.names, ("docs-writer", "test-skill"))
        self.assertEqual(selection.unknown, ("missing-skill",))

    def test_activation_adds_context_fragment_and_lists_resources(self):
        cfg = self.make_cfg()
        skill_root = cfg.root / ".agents" / "skills" / "docs-writer"
        write_skill(skill_root, "docs-writer")
        (skill_root / "references").mkdir()
        (skill_root / "references" / "STYLE.md").write_text("Style", encoding="utf-8")
        index = discover_skills(cfg)
        runtime = SkillRuntime(cfg, index)

        obs = runtime.activate("docs-writer", "test")

        self.assertTrue(obs["ok"])
        self.assertEqual(obs["resources"][0]["path"], "references/STYLE.md")
        fragments = obs["_context_fragments"]
        self.assertIn("# Active Agent Skill: docs-writer", fragments[0].text)
        self.assertIn("workspace path:", fragments[0].text)

    def test_list_skill_resources_rejects_symlink_escape(self):
        cfg = self.make_cfg()
        skill_root = cfg.root / ".agents" / "skills" / "docs-writer"
        write_skill(skill_root, "docs-writer")
        outside = cfg.root / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (skill_root / "references").mkdir()
        (skill_root / "references" / "outside.txt").symlink_to(outside)
        skill = discover_skills(cfg).skills["docs-writer"]

        resources = list_skill_resources(skill, cfg.root)

        self.assertEqual(resources, [])

    def test_run_explicit_marker_activates_skill_before_first_model_call(self):
        cfg = self.make_cfg()
        write_skill(cfg.root / ".agents" / "skills" / "docs-writer", "docs-writer")
        seen = []

        def fake_chat(messages, tools=None):
            seen.append((json.loads(json.dumps(messages)), json.loads(json.dumps(tools))))
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("@skill:docs-writer обнови README", cfg)

        self.assertEqual(result, "done")
        system = seen[0][0][0]["content"]
        self.assertIn("# Active Agent Skill: docs-writer", system)
        self.assertNotIn("# Available Agent Skills", system)
        tool_names = [item["function"]["name"] for item in seen[0][1]]
        self.assertNotIn("activate_skill", tool_names)
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        activated = [item for item in events if item["event"] == "skill_activated"]
        self.assertEqual(activated[0]["trigger"], "explicit")

    def test_run_model_can_activate_skill_from_catalog(self):
        cfg = self.make_cfg()
        write_skill(cfg.root / ".agents" / "skills" / "docs-writer", "docs-writer")
        seen = []

        def fake_chat(messages, tools=None):
            seen.append((json.loads(json.dumps(messages)), json.loads(json.dumps(tools))))
            if len(seen) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_skill",
                                        "function": {
                                            "name": "activate_skill",
                                            "arguments": json.dumps({"name": "docs-writer"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("обнови README", cfg)

        self.assertEqual(result, "done")
        first_system = seen[0][0][0]["content"]
        second_system = seen[1][0][0]["content"]
        self.assertIn("# Available Agent Skills", first_system)
        self.assertIn("# Active Agent Skill: docs-writer", second_system)
        activate_schema = next(
            item["function"]
            for item in seen[0][1]
            if item["function"]["name"] == "activate_skill"
        )
        self.assertEqual(
            activate_schema["parameters"]["properties"]["name"]["enum"],
            ["docs-writer"],
        )
        trace_text = Path(trace_path).read_text(encoding="utf-8")
        self.assertNotIn("Пиши коротко и проверяемо.", trace_text)

    def test_run_unknown_explicit_skill_fails_before_model_call(self):
        cfg = self.make_cfg()

        with patch("madharness_mini.loop.ModelClient.chat") as chat:
            with self.assertRaisesRegex(RuntimeError, "unknown skill: docs-writer"):
                run_agent("@skill:docs-writer обнови README", cfg)

        chat.assert_not_called()

    def test_ask_does_not_load_skill_catalog_or_parse_markers(self):
        cfg = self.make_cfg()
        write_skill(cfg.root / ".agents" / "skills" / "docs-writer", "docs-writer")
        seen = []

        def fake_chat(messages, tools=None):
            seen.append(json.loads(json.dumps(messages)))
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, _trace = ask("@skill:missing обычный вопрос", cfg)

        self.assertEqual(result, "ok")
        rendered = json.dumps(seen[0], ensure_ascii=False)
        self.assertNotIn("Available Agent Skills", rendered)
        self.assertNotIn("Active Agent Skill", rendered)

    def test_skills_cli_list_show_validate(self):
        cfg = self.make_cfg()
        write_skill(cfg.root / ".agents" / "skills" / "docs-writer", "docs-writer")

        listing = skills_command(cfg, "list")
        shown = skills_command(cfg, "show", "docs-writer")
        validated = skills_command(cfg, "validate")

        self.assertIn("docs-writer", listing)
        self.assertIn("instructions:", shown)
        self.assertIn("OK", validated)
