import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from madharness_mini.hooks import HookManager
from madharness_mini.loop import ask, run_agent
from madharness_mini.trace import Trace

from tests.helpers import HarnessTestCase


class HookTests(HarnessTestCase):
    def write_hook_config(self, cfg, *hooks):
        (cfg.state_dir / "hooks.json").write_text(
            json.dumps({"hooks": list(hooks)}),
            encoding="utf-8",
        )

    def test_missing_hooks_config_is_noop(self):
        cfg = self.make_cfg()
        trace = Trace(cfg, "run")
        manager = HookManager.from_config(cfg, trace)

        decision = manager.emit("session_start", kind="run", data={"task": "hello"})

        self.assertTrue(decision.ok)
        events = [
            json.loads(line)
            for line in Path(trace.path).read_text(encoding="utf-8").splitlines()
        ]
        self.assertNotIn("hook_started", [event["event"] for event in events])

    def test_before_tool_call_hook_blocks_before_handler(self):
        cfg = self.make_cfg()
        hook = cfg.root / "deny_shell.py"
        hook.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "event = json.load(sys.stdin)",
                    "assert event['event'] == 'before_tool_call'",
                    "assert event['data']['tool'] == 'run_shell'",
                    "print(json.dumps({'ok': False, 'block': 'no shell'}))",
                ]
            ),
            encoding="utf-8",
        )
        self.write_hook_config(
            cfg,
            {
                "id": "deny-shell",
                "event": "before_tool_call",
                "match": {"tool": "run_shell"},
                "command": sys.executable,
                "args": [str(hook)],
                "cwd": ".",
                "timeout_seconds": 3,
            },
        )
        seen_messages = []

        def fake_chat(messages, tools=None):
            seen_messages.append(json.loads(json.dumps(messages)))
            if len(seen_messages) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_shell",
                                        "function": {
                                            "name": "run_shell",
                                            "arguments": json.dumps({"command": "pwd"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

        with (
            patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat),
            patch("madharness_mini.tools.registry.ToolRegistry.call") as tool_call,
        ):
            result, trace_path = run_agent("run pwd", cfg)

        self.assertEqual(result, "done")
        tool_call.assert_not_called()
        second_request = json.dumps(seen_messages[1], ensure_ascii=False)
        self.assertIn("blocked by hook: no shell", second_request)
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        self.assertIn("hook_blocked", [event["event"] for event in events])
        observation = next(
            event["observation"]
            for event in events
            if event["event"] == "tool_observation"
        )
        self.assertFalse(observation["ok"])
        self.assertTrue(observation["hook_blocked"])

    def test_hook_process_failure_is_traced_without_breaking_ask(self):
        cfg = self.make_cfg()
        hook = cfg.root / "broken_hook.py"
        hook.write_text(
            "import sys\nsys.stderr.write('boom')\nsys.exit(2)\n",
            encoding="utf-8",
        )
        self.write_hook_config(
            cfg,
            {
                "id": "broken",
                "event": "before_model_call",
                "command": sys.executable,
                "args": [str(hook)],
                "cwd": ".",
                "timeout_seconds": 3,
            },
        )

        with patch(
            "madharness_mini.loop.ModelClient.chat",
            return_value={"choices": [{"message": {"content": "ok"}}]},
        ):
            result, trace_path = ask("hello", cfg)

        self.assertEqual(result, "ok")
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        failed = next(event for event in events if event["event"] == "hook_failed")
        self.assertEqual(failed["hook"], "broken")
        self.assertIn("boom", failed["error"])

    def test_command_hook_does_not_inherit_madharness_env(self):
        cfg = self.make_cfg()
        hook = cfg.root / "check_env.py"
        hook.write_text(
            "\n".join(
                [
                    "import json, os",
                    "if os.environ.get('MADHARNESS_MINI_API_KEY'):",
                    "    print(json.dumps({'ok': False, 'block': 'env leaked'}))",
                    "else:",
                    "    print(json.dumps({'ok': True, 'message': 'clean'}))",
                ]
            ),
            encoding="utf-8",
        )
        self.write_hook_config(
            cfg,
            {
                "id": "env-check",
                "event": "session_start",
                "command": sys.executable,
                "args": [str(hook)],
                "cwd": ".",
                "timeout_seconds": 3,
            },
        )
        trace = Trace(cfg, "run")
        manager = HookManager.from_config(cfg, trace)

        with patch.dict(os.environ, {"MADHARNESS_MINI_API_KEY": "secret"}):
            decision = manager.emit("session_start", kind="run", data={})

        self.assertTrue(decision.ok)
        events = [
            json.loads(line)
            for line in Path(trace.path).read_text(encoding="utf-8").splitlines()
        ]
        finished = next(event for event in events if event["event"] == "hook_finished")
        self.assertEqual(finished["hook"], "env-check")
        self.assertEqual(finished["message"], "clean")
