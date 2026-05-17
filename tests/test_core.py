import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from madharness_mini.cli import main
from madharness_mini.config import Config
from madharness_mini.instructions import load_agents_md, load_prompt
from madharness_mini.loop import base_messages
from madharness_mini.model import ModelClient
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
            {"function": {"name": "read_file", "arguments": '{"path":"AGENTS.md"}'}}
        )
        self.assertEqual(name, "read_file")
        self.assertEqual(args["path"], "AGENTS.md")

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

    def test_load_agents_md_from_root_to_nested_dir(self):
        cfg = self.make_cfg()
        nested = cfg.root / "pkg" / "sub"
        nested.mkdir(parents=True)
        (cfg.root / "AGENTS.md").write_text("root rules", encoding="utf-8")
        (cfg.root / "pkg" / "AGENTS.md").write_text("pkg rules", encoding="utf-8")
        text = load_agents_md(cfg.root, nested)
        self.assertLess(text.index("root rules"), text.index("pkg rules"))

    def test_load_agents_md_applies_one_combined_limit(self):
        cfg = self.make_cfg()
        nested = cfg.root / "pkg" / "sub"
        nested.mkdir(parents=True)
        root_text = "root rules " + ("x" * 100) + " root tail"
        (cfg.root / "AGENTS.md").write_text(root_text, encoding="utf-8")
        (cfg.root / "pkg" / "AGENTS.md").write_text("pkg rules", encoding="utf-8")
        root_chunk = f"# Instructions from {cfg.root / 'AGENTS.md'}\n{root_text}"

        with patch(
            "madharness_mini.instructions.Path.home", return_value=cfg.root / "home"
        ):
            text = load_agents_md(
                cfg.root, nested, max_bytes=len(root_chunk.encode("utf-8")) + 5
            )

        self.assertIn("root tail", text)
        self.assertNotIn("pkg rules", text)
        self.assertIn("...[clipped", text)

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

    def test_base_messages_orders_system_prompt_before_agents_md(self):
        cfg = self.make_cfg()
        nested = cfg.root / "pkg" / "sub"
        nested.mkdir(parents=True)
        cfg.cwd = nested
        (cfg.root / "AGENTS.md").write_text("root rules", encoding="utf-8")
        (cfg.root / "pkg" / "AGENTS.md").write_text("pkg rules", encoding="utf-8")

        system = base_messages(cfg, "task")[0]["content"]

        self.assertLess(system.index(load_prompt("system")), system.index("root rules"))
        self.assertLess(system.index("root rules"), system.index("pkg rules"))


if __name__ == "__main__":
    unittest.main()
