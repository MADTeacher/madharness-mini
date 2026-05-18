import json
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from madharness_mini.cli import main
from madharness_mini.config import Config
from madharness_mini.model import ModelClient

from tests.helpers import HarnessTestCase


class ConfigCliTests(HarnessTestCase):
    def test_config_defaults_merge_with_file(self):
        cfg = self.make_cfg()
        self.assertEqual(cfg.data["base_url"], "https://openrouter.ai/api/v1")
        self.assertNotIn("provider", cfg.data)
        self.assertNotIn("providers", cfg.data)
        self.assertTrue(cfg.data["allow_shell"])

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
