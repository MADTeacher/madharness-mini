import json
import os
import tempfile
import urllib.error
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch

from madharness_mini.cli import main
from madharness_mini.config import Config
from madharness_mini.instructions import (
    PROJECT_DOC_MAX_BYTES,
    load_project_instructions,
    load_prompt,
)
from madharness_mini.loop import ask, base_messages
from madharness_mini.model import ModelClient, ModelRateLimitError, parse_retry_after
from madharness_mini.policy import Policy
from madharness_mini.tools import ToolRegistry
from madharness_mini.utils import fail, ok, parse_tool_args


class CoreTests(unittest.TestCase):
    def make_cfg(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text(
            json.dumps({"workspace_root": ".", "allow_shell": True}),
            encoding="utf-8",
        )
        cfg = Config(root)
        self.addCleanup(tmp.cleanup)
        return cfg

    def test_config_defaults_merge_with_file(self):
        cfg = self.make_cfg()
        self.assertEqual(cfg.data["base_url"], "https://openrouter.ai/api/v1")
        self.assertNotIn("provider", cfg.data)
        self.assertNotIn("providers", cfg.data)
        self.assertTrue(cfg.data["allow_shell"])

    def test_model_client_reads_top_level_base_url(self):
        cfg = self.make_cfg()
        cfg.data["base_url"] = "https://kodikrouter.ru/api/v1"
        settings = ModelClient(cfg).settings()
        self.assertEqual(settings["base_url"], "https://kodikrouter.ru/api/v1")

    def test_model_client_reads_top_level_api_key(self):
        cfg = self.make_cfg()
        cfg.data["base_url"] = "http://localhost:9999/v1"
        cfg.data["api_key"] = "token"
        settings = ModelClient(cfg).settings()
        self.assertEqual(settings["base_url"], "http://localhost:9999/v1")
        self.assertEqual(settings["api_key"], "token")

    def test_config_ignores_legacy_provider_fields(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text(
            json.dumps(
                {
                    "provider": "kodikrouter",
                    "providers": {
                        "kodikrouter": {
                            "base_url": "https://llm.example.test/api/v1",
                            "api_key": "",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        cfg = Config(root)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(cfg.data["base_url"], "https://openrouter.ai/api/v1")
        self.assertNotIn("provider", cfg.data)
        self.assertNotIn("providers", cfg.data)

    def test_env_file_overrides_config(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text(
            json.dumps({"model": "old", "base_url": "https://old.example/v1"}),
            encoding="utf-8",
        )
        (root / ".env").write_text(
            "MADHARNESS_MINI_BASE_URL=https://new.example/v1\nMADHARNESS_MINI_MODEL=deepseek/deepseek-v4-flash\nMADHARNESS_MINI_API_KEY=secret\n",
            encoding="utf-8",
        )
        cfg = Config(root)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(cfg.data["base_url"], "https://new.example/v1")
        self.assertEqual(cfg.data["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(ModelClient(cfg).settings()["api_key"], "secret")

    def test_init_command_creates_config_with_api_key(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        old_cwd = os.getcwd()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(root)
        out = StringIO()

        with redirect_stdout(out):
            main(
                [
                    "init",
                    "--base-url",
                    "https://kodikrouter.ru/api/v1",
                    "--model",
                    "deepseek/deepseek-v4-flash",
                    "--api-key",
                    "secret",
                    "--no-prompt",
                ]
            )

        path = root / ".madharness-mini" / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("provider", data)
        self.assertNotIn("providers", data)
        self.assertEqual(data["base_url"], "https://kodikrouter.ru/api/v1")
        self.assertEqual(data["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(data["api_key"], "secret")
        self.assertIn("Настройка записана", out.getvalue())

    def test_init_command_warns_when_api_key_missing(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        old_cwd = os.getcwd()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(root)
        out = StringIO()

        with patch.dict(os.environ, {}, clear=True), redirect_stdout(out):
            main(["init", "--no-prompt"])

        data = json.loads(
            (root / ".madharness-mini" / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data["api_key"], "")
        self.assertIn("Ключ API не задан", out.getvalue())

    def test_init_command_prompts_with_default_router_model_and_config(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        old_cwd = os.getcwd()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(root)
        out = StringIO()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("madharness_mini.cli.sys.stdin.isatty", return_value=True),
            patch("madharness_mini.cli.getpass.getpass", return_value="secret") as prompt,
            redirect_stdout(out),
        ):
            main(["init"])

        prompt_text = prompt.call_args.args[0]
        self.assertIn("OpenRouter", prompt_text)
        self.assertIn("deepseek/deepseek-v4-flash", prompt_text)
        self.assertIn(".madharness-mini/config.json", prompt_text)

        data = json.loads(
            (root / ".madharness-mini" / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data["api_key"], "secret")

    def test_model_client_mentions_init_when_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = self.make_cfg()
        with self.assertRaisesRegex(RuntimeError, "madharness-mini init"):
            ModelClient(cfg).chat([{"role": "user", "content": "hello"}])

    def test_parse_retry_after_seconds(self):
        self.assertEqual(parse_retry_after("7"), 7)

    def test_parse_retry_after_http_date(self):
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        seconds = parse_retry_after(format_datetime(retry_at, usegmt=True))

        self.assertIsNotNone(seconds)
        self.assertGreaterEqual(seconds, 1)
        self.assertLessEqual(seconds, 30)

    def test_parse_retry_after_missing_or_invalid(self):
        self.assertIsNone(parse_retry_after(None))
        self.assertIsNone(parse_retry_after(""))
        self.assertIsNone(parse_retry_after("not a date"))

    def test_model_client_raises_rate_limit_error_for_http_429(self):
        cfg = self.make_cfg()
        cfg.data["api_key"] = "token"
        err = urllib.error.HTTPError(
            url="https://llm.example.test/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "3"},
            fp=BytesIO(b'{"error":"limited"}'),
        )

        with patch("madharness_mini.model.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(ModelRateLimitError) as caught:
                ModelClient(cfg).chat([{"role": "user", "content": "hello"}])

        exc = caught.exception
        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.body, '{"error":"limited"}')
        self.assertEqual(exc.retry_after, "3")
        self.assertEqual(exc.retry_after_seconds, 3)

    def test_ask_retries_once_after_short_rate_limit(self):
        cfg = self.make_cfg()
        rate_limit = ModelRateLimitError(
            status=429,
            body="limited",
            retry_after="1",
            retry_after_seconds=1,
        )
        raw = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch("madharness_mini.loop.ModelClient.chat", side_effect=[rate_limit, raw]),
            patch("madharness_mini.loop.time.sleep") as sleep,
        ):
            result, trace_path = ask("hello", cfg)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(1)
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        self.assertIn("model_rate_limit_retry", [event["event"] for event in events])

    def test_ask_fails_when_rate_limit_retry_after_is_too_long(self):
        cfg = self.make_cfg()
        rate_limit = ModelRateLimitError(
            status=429,
            body="limited",
            retry_after="61",
            retry_after_seconds=61,
        )

        with (
            patch("madharness_mini.loop.ModelClient.chat", side_effect=rate_limit),
            patch("madharness_mini.loop.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "достигнут лимит LLM API"):
                ask("hello", cfg)

        sleep.assert_not_called()

    def test_ask_fails_when_retry_hits_rate_limit_again(self):
        cfg = self.make_cfg()
        first = ModelRateLimitError(
            status=429,
            body="limited",
            retry_after="1",
            retry_after_seconds=1,
        )
        second = ModelRateLimitError(
            status=429,
            body="still limited",
            retry_after="1",
            retry_after_seconds=1,
        )

        with (
            patch("madharness_mini.loop.ModelClient.chat", side_effect=[first, second]),
            patch("madharness_mini.loop.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "достигнут лимит LLM API"):
                ask("hello", cfg)

        sleep.assert_called_once_with(1)

    def test_policy_denies_outside_workspace(self):
        policy = Policy(self.make_cfg())
        path, err = policy.safe_path("../outside.txt")
        self.assertIsNone(path)
        self.assertIn("outside workspace", err)

    def test_policy_denies_protected_paths(self):
        policy = Policy(self.make_cfg())
        path, err = policy.safe_path(".git/config")
        self.assertIsNone(path)
        self.assertIn("protected path", err)

    def test_shell_policy_denies_risky_commands(self):
        policy = Policy(self.make_cfg())
        self.assertFalse(policy.shell_allowed("rm -rf .")[0])
        self.assertFalse(policy.shell_allowed("curl https://example.com")[0])
        self.assertTrue(policy.shell_allowed("uv run -m unittest discover -s tests")[0])

    def test_observation_format(self):
        self.assertEqual(ok("x", "done")["ok"], True)
        self.assertEqual(fail("x", "bad")["ok"], False)

    def test_parse_tool_args(self):
        name, args = parse_tool_args(
            {"function": {"name": "read_file", "arguments": '{"path":"hello.txt"}'}}
        )
        self.assertEqual(name, "read_file")
        self.assertEqual(args["path"], "hello.txt")

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

    def test_apply_patch_is_not_registered(self):
        schemas = ToolRegistry(self.make_cfg()).schemas()
        names = [item["function"]["name"] for item in schemas]
        self.assertNotIn("apply_patch", names)

    def test_apply_patch_call_returns_unknown_tool(self):
        obs = ToolRegistry(self.make_cfg()).call("apply_patch", {"patch": ""})
        self.assertFalse(obs["ok"])
        self.assertIn("unknown tool", obs["summary"])

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


if __name__ == "__main__":
    unittest.main()
