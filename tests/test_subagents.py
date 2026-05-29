import json
from pathlib import Path
from unittest.mock import patch

from madharness_mini.cli import subagents_command
from madharness_mini.loop import run_agent
from madharness_mini.subagents import discover_subagents

from tests.helpers import HarnessTestCase


def write_subagent(
    root: Path,
    name: str,
    *,
    profile: str = "writable",
    tools: list[str] | None = None,
    override: bool = False,
) -> None:
    """Пишем project-local markdown-субагента для тестов discovery и run."""

    root.mkdir(parents=True, exist_ok=True)
    tools = tools or ["list_files", "read_file", "search_code", "write_file"]
    lines = [
        "---",
        f"name: {name}",
        f"description: Test subagent {name}.",
        f"profile: {profile}",
        "tools: " + json.dumps(tools),
        "max_turns: 4",
    ]
    if override:
        lines.append("override: true")
    lines.extend(
        [
            "---",
            "",
            f"Ты тестовый субагент {name}.",
            "Верни короткий итог.",
        ]
    )
    (root / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


class SubagentTests(HarnessTestCase):
    def tool_names(self, tools):
        return [item["function"]["name"] for item in tools or []]

    def test_discover_subagents_reads_builtin_and_project_markdown(self):
        cfg = self.make_cfg()
        write_subagent(cfg.root / ".madharness-mini" / "subagents", "test-writer")

        index = discover_subagents(cfg)

        self.assertIn("planner", index.names())
        self.assertIn("test-writer", index.names())
        subagent = index.subagents["test-writer"]
        self.assertEqual(subagent.source, "project")
        self.assertEqual(subagent.tools, ("list_files", "read_file", "search_code", "write_file"))

    def test_project_subagent_cannot_shadow_builtin_without_override(self):
        cfg = self.make_cfg()
        write_subagent(cfg.root / ".madharness-mini" / "subagents", "planner")

        index = discover_subagents(cfg)

        self.assertEqual(index.subagents["planner"].source, "builtin")
        self.assertTrue(
            any("without override" in item.message for item in index.diagnostics)
        )

    def test_subagent_tools_must_be_json_style_list(self):
        cfg = self.make_cfg()
        root = cfg.root / ".madharness-mini" / "subagents"
        root.mkdir(parents=True)
        (root / "bad-tools.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: bad-tools",
                    "description: Bad tools format.",
                    "profile: writable",
                    "tools: list_files read_file",
                    "---",
                    "body",
                ]
            ),
            encoding="utf-8",
        )

        index = discover_subagents(cfg)

        self.assertNotIn("bad-tools", index.subagents)
        self.assertTrue(
            any("JSON-style list" in item.message for item in index.diagnostics)
        )

    def test_subagents_cli_list_show_validate(self):
        cfg = self.make_cfg()
        write_subagent(cfg.root / ".madharness-mini" / "subagents", "test-writer")

        listing = subagents_command(cfg, "list")
        shown = subagents_command(cfg, "show", "test-writer")
        validated = subagents_command(cfg, "validate")

        self.assertIn("test-writer", listing)
        self.assertIn("prompt:", shown)
        self.assertIn("OK", validated)

    def test_delegate_task_runs_writable_subagent_with_local_trace(self):
        cfg = self.make_cfg()
        write_subagent(cfg.root / ".madharness-mini" / "subagents", "test-writer")
        seen_tool_names = []

        def fake_chat(messages, tools=None):
            names = self.tool_names(tools)
            seen_tool_names.append(names)
            if len(seen_tool_names) == 1:
                self.assertIn("delegate_task", names)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_delegate",
                                        "function": {
                                            "name": "delegate_task",
                                            "arguments": json.dumps(
                                                {
                                                    "subagent": "test-writer",
                                                    "task": "write result",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if len(seen_tool_names) == 2 and "write_file" in names:
                self.assertNotIn("delegate_task", names)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_write",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": json.dumps(
                                                {
                                                    "path": "subagent-result.txt",
                                                    "content": "done\n",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if len(seen_tool_names) == 3:
                return {"choices": [{"message": {"content": "subagent done"}}]}
            return {"choices": [{"message": {"content": "parent done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("delegate work", cfg)

        self.assertEqual(result, "parent done")
        self.assertEqual(
            (cfg.root / "subagent-result.txt").read_text(encoding="utf-8"),
            "done\n",
        )
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        finished = next(item for item in events if item["event"] == "subagent_finished")
        child_path = cfg.cwd / finished["trace_path"]
        self.assertTrue(child_path.is_file())
        delegate_obs = next(
            item["observation"]
            for item in events
            if item.get("event") == "tool_observation" and item.get("tool") == "delegate_task"
        )
        self.assertEqual(delegate_obs["subagent"], "test-writer")
        self.assertEqual(delegate_obs["changed_files"], ["subagent-result.txt"])
        self.assertEqual(delegate_obs["subagent_trace_id"], finished["trace_id"])

    def test_orchestration_off_hides_delegate_task(self):
        cfg = self.make_cfg()
        seen_tool_names = []

        def fake_chat(messages, tools=None):
            names = self.tool_names(tools)
            seen_tool_names.append(names)
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent(
                "обычная маленькая задача",
                cfg,
                orchestration_mode="off",
            )

        self.assertEqual(result, "done")
        self.assertNotIn("delegate_task", seen_tool_names[0])
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        mode = next(item for item in events if item["event"] == "orchestration_mode")
        self.assertEqual(mode["configured"], "off")
        self.assertEqual(mode["effective"], "off")

    def test_requested_orchestration_only_appears_when_task_asks(self):
        cfg = self.make_cfg()
        seen_tool_names = []

        def fake_chat(messages, tools=None):
            names = self.tool_names(tools)
            seen_tool_names.append(names)
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            run_agent("обычная маленькая задача", cfg, orchestration_mode="requested")
            run_agent(
                "используй субагентов для проверки",
                cfg,
                orchestration_mode="requested",
            )

        self.assertNotIn("delegate_task", seen_tool_names[0])
        self.assertIn("delegate_task", seen_tool_names[1])

    def test_required_orchestration_limits_parent_tools(self):
        cfg = self.make_cfg()
        seen_tool_names = []

        def fake_chat(messages, tools=None):
            names = self.tool_names(tools)
            seen_tool_names.append(names)
            system_text = "\n".join(
                message.get("content") or ""
                for message in messages
                if message.get("role") == "system"
            )
            self.assertIn("Оркестрация обязательна", system_text)
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent(
                "длинная задача",
                cfg,
                orchestration_mode="required",
            )

        self.assertEqual(result, "done")
        self.assertEqual(
            set(seen_tool_names[0]),
            {"list_files", "read_file", "search_code", "delegate_task"},
        )
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        mode = next(item for item in events if item["event"] == "orchestration_mode")
        self.assertEqual(mode["effective"], "required")

    def test_planner_can_request_user_input_through_delegate_task(self):
        cfg = self.make_cfg()
        seen_tool_names = []

        def fake_chat(messages, tools=None):
            names = self.tool_names(tools)
            seen_tool_names.append(names)
            if len(seen_tool_names) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_delegate",
                                        "function": {
                                            "name": "delegate_task",
                                            "arguments": json.dumps(
                                                {
                                                    "subagent": "planner",
                                                    "task": "choose path",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if "ask_user" in names:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_question",
                                        "function": {
                                            "name": "ask_user",
                                            "arguments": json.dumps(
                                                {
                                                    "question": "Какой путь выбрать?",
                                                    "options": ["A", "B"],
                                                    "reason": "Нужно решение.",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "parent asks user"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("plan with question", cfg)

        self.assertIn("planner просит уточнение", result)
        self.assertIn("Какой путь выбрать?", result)
        self.assertIn("1. A", result)
        self.assertIn("2. B", result)
        self.assertIn("Причина: Нужно решение.", result)
        self.assertEqual(len(seen_tool_names), 2)
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        delegate_obs = next(
            item["observation"]
            for item in events
            if item.get("event") == "tool_observation" and item.get("tool") == "delegate_task"
        )
        self.assertEqual(delegate_obs["status"], "needs_user_input")
        self.assertEqual(delegate_obs["question"], "Какой путь выбрать?")
        self.assertIn("subagent_trace_id", delegate_obs)
        requested = next(
            item for item in events if item.get("event") == "user_input_requested"
        )
        self.assertEqual(requested["question"], "Какой путь выбрать?")

    def test_planner_cannot_write_non_markdown_files(self):
        cfg = self.make_cfg()
        seen_tool_names = []

        def fake_chat(messages, tools=None):
            names = self.tool_names(tools)
            seen_tool_names.append(names)
            if len(seen_tool_names) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_delegate",
                                        "function": {
                                            "name": "delegate_task",
                                            "arguments": json.dumps(
                                                {
                                                    "subagent": "planner",
                                                    "task": "create index",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if len(seen_tool_names) == 2 and "write_file" in names:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_write",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": json.dumps(
                                                {
                                                    "path": "index.html",
                                                    "content": "<html></html>\n",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if len(seen_tool_names) == 3:
                return {"choices": [{"message": {"content": "planner stopped"}}]}
            return {"choices": [{"message": {"content": "parent done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("planner writes index", cfg)

        self.assertEqual(result, "parent done")
        self.assertFalse((cfg.root / "index.html").exists())
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        delegate_obs = next(
            item["observation"]
            for item in events
            if item.get("event") == "tool_observation" and item.get("tool") == "delegate_task"
        )
        child_path = cfg.cwd / delegate_obs["subagent_trace_path"]
        child_events = [
            json.loads(line)
            for line in child_path.read_text(encoding="utf-8").splitlines()
        ]
        write_obs = next(
            item["observation"]
            for item in child_events
            if item.get("event") == "tool_observation" and item.get("tool") == "write_file"
        )
        self.assertFalse(write_obs["ok"])
        self.assertIn("planner may write only Markdown plan files", write_obs["summary"])
