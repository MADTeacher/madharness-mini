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
        # Shell-режим с интерпретатором — новый дефолт; фон-процессы ограничены.
        self.assertIs(cfg.data["allow_shell_interpreter"], True)
        self.assertEqual(cfg.data["max_background_processes"], 4)

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
        # Чистим окружение процесса, чтобы тест проверял только файл `.env`.
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config(root)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(cfg.data["base_url"], "https://new.example/v1")
        self.assertEqual(cfg.data["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(ModelClient(cfg).settings()["api_key"], "secret")

    def test_env_file_overrides_image_settings_with_types(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text("{}", encoding="utf-8")
        (root / ".env").write_text(
            "MADHARNESS_MINI_SUPPORTS_IMAGE_INPUT=true\n"
            "MADHARNESS_MINI_MAX_IMAGE_BYTES=42\n"
            "MADHARNESS_MINI_IMAGE_DETAIL=high\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            cfg = Config(root)

        self.addCleanup(tmp.cleanup)
        self.assertIs(cfg.data["supports_image_input"], True)
        self.assertEqual(cfg.data["max_image_bytes"], 42)
        self.assertEqual(cfg.data["image_detail"], "high")

    def test_env_file_overrides_shell_settings(self):
        # allow_shell, allow_shell_interpreter и max_background_processes
        # читаются из .env с приведением типов (bool/int).
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text("{}", encoding="utf-8")
        (root / ".env").write_text(
            "MADHARNESS_MINI_ALLOW_SHELL=false\n"
            "MADHARNESS_MINI_ALLOW_SHELL_INTERPRETER=false\n"
            "MADHARNESS_MINI_MAX_BACKGROUND_PROCESSES=8\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            cfg = Config(root)

        self.addCleanup(tmp.cleanup)
        self.assertIs(cfg.data["allow_shell"], False)
        self.assertIs(cfg.data["allow_shell_interpreter"], False)
        self.assertEqual(cfg.data["max_background_processes"], 8)

    def test_env_file_overrides_orchestration_settings(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text("{}", encoding="utf-8")
        (root / ".env").write_text(
            "MADHARNESS_MINI_ORCHESTRATION_MODE=requested\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            cfg = Config(root)

        self.addCleanup(tmp.cleanup)
        self.assertNotIn("orchestration_enabled", cfg.data)
        self.assertEqual(cfg.data["orchestration_mode"], "requested")

    def test_env_file_disables_orchestration_via_mode(self):
        # Полностью выключить делегирование теперь можно только через mode=off,
        # без отдельного булева выключателя.
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text("{}", encoding="utf-8")
        (root / ".env").write_text(
            "MADHARNESS_MINI_ORCHESTRATION_MODE=off\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            cfg = Config(root)

        self.addCleanup(tmp.cleanup)
        self.assertEqual(cfg.data["orchestration_mode"], "off")

    def test_env_file_rejects_unknown_orchestration_mode(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text("{}", encoding="utf-8")
        (root / ".env").write_text(
            "MADHARNESS_MINI_ORCHESTRATION_MODE=always\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "MADHARNESS_MINI_ORCHESTRATION_MODE",
            ):
                Config(root)

        self.addCleanup(tmp.cleanup)

    def test_env_file_rejects_unknown_image_detail(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / ".madharness-mini").mkdir()
        (root / ".madharness-mini" / "config.json").write_text("{}", encoding="utf-8")
        (root / ".env").write_text(
            "MADHARNESS_MINI_IMAGE_DETAIL=microscope\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MADHARNESS_MINI_IMAGE_DETAIL"):
                Config(root)

        self.addCleanup(tmp.cleanup)

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

    def test_run_command_passes_orchestration_flag(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        old_cwd = os.getcwd()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(root)
        out = StringIO()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("madharness_mini.cli.run_agent", return_value=("ok", "trace.jsonl")) as run,
            redirect_stdout(out),
        ):
            main(["run", "--orchestrate-required", "сделай проект"])

        self.assertEqual(run.call_args.kwargs["orchestration_mode"], "required")
        self.assertIn("ok", out.getvalue())

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

    def test_ensure_dirs_creates_and_sweeps_tool_outputs(self):
        """ensure_dirs создаёт tool_outputs и подчищает файлы старше TTL дней."""
        import time

        from madharness_mini.config import TOOL_OUTPUTS_TTL_DAYS

        cfg = self.make_cfg()
        outputs = cfg.state_dir / "tool_outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        # Старый файл: backdate mtime за границу TTL.
        stale = outputs / "shell-stale.log"
        stale.write_text("old", encoding="utf-8")
        old_mtime = time.time() - (TOOL_OUTPUTS_TTL_DAYS + 1) * 24 * 3600
        os.utime(stale, (old_mtime, old_mtime))
        # Свежий файл: должен остаться.
        fresh = outputs / "shell-fresh.log"
        fresh.write_text("new", encoding="utf-8")

        cfg.ensure_dirs()

        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())

    def test_default_config_enables_shell_full_output(self):
        """По умолчанию полный вывод shell сохраняется во временный файл."""
        cfg = self.make_cfg()
        self.assertIs(cfg.data["context_shell_full_output"], True)
