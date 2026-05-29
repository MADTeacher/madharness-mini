import json
import os
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from madharness_mini.loop import ask, run_agent
from madharness_mini.model import ModelClient, ModelRateLimitError, parse_retry_after
from madharness_mini.trace import summarize_trace

from tests.helpers import HarnessTestCase

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


class ModelLoopTests(HarnessTestCase):
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

    def test_ask_writes_context_report_to_trace(self):
        cfg = self.make_cfg()
        raw = {"choices": [{"message": {"content": "ok"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", return_value=raw):
            result, trace_path = ask("hello", cfg)

        self.assertEqual(result, "ok")
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        started = next(event for event in events if event["event"] == "model_call_started")
        report = started["context_report"]
        self.assertIsInstance(report["request_tokens_estimate"], int)
        self.assertEqual(report["tools_tokens_estimate"], 0)
        self.assertEqual(report["history"]["total_entries"], 0)
        summary = summarize_trace(cfg, Path(trace_path).stem)
        self.assertIn("context:", summary)
        self.assertIn("estimated tokens", summary)
        self.assertIn("history: 0/0 entries", summary)

    def test_run_agent_keeps_image_text_only_when_vision_is_disabled(self):
        cfg = self.make_cfg()
        (cfg.root / "shot.png").write_bytes(PNG_BYTES)
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
                                        "id": "call_1",
                                        "function": {
                                            "name": "read_image",
                                            "arguments": json.dumps({"path": "shot.png"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("inspect screenshot", cfg)

        self.assertEqual(result, "done")
        second_request = json.dumps(seen_messages[1])
        self.assertNotIn("data:image", second_request)
        self.assertNotIn("base64", second_request)
        trace_text = Path(trace_path).read_text(encoding="utf-8")
        self.assertNotIn("data:image", trace_text)
        self.assertNotIn("base64", trace_text)

    def test_run_agent_attaches_image_when_vision_is_enabled(self):
        cfg = self.make_cfg()
        cfg.data["supports_image_input"] = True
        (cfg.root / "shot.png").write_bytes(PNG_BYTES)
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
                                        "id": "call_1",
                                        "function": {
                                            "name": "read_image",
                                            "arguments": json.dumps({"path": "shot.png"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("inspect screenshot", cfg)

        self.assertEqual(result, "done")
        image_messages = [
            message
            for message in seen_messages[1]
            if message.get("role") == "user" and isinstance(message.get("content"), list)
        ]
        self.assertEqual(len(image_messages), 1)
        image_part = image_messages[0]["content"][1]
        self.assertEqual(image_part["type"], "image_url")
        self.assertEqual(image_part["image_url"]["detail"], "auto")
        self.assertIn("data:image/png;base64,", image_part["image_url"]["url"])
        trace_text = Path(trace_path).read_text(encoding="utf-8")
        self.assertNotIn("data:image", trace_text)
        self.assertNotIn("base64", trace_text)

    def test_run_agent_trims_large_tool_output_before_next_model_call(self):
        cfg = self.make_cfg()
        cfg.data["context_max_tokens"] = 4000
        seen_messages = []
        huge_stdout = "x" * 5000

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
                                        "id": "call_1",
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
            patch(
                "madharness_mini.loop.ToolRegistry.call",
                return_value={
                    "ok": True,
                    "tool": "run_shell",
                    "summary": "ran pwd",
                    "stdout": huge_stdout,
                    "stderr": "",
                },
            ),
        ):
            result, _trace_path = run_agent("run command", cfg)

        self.assertEqual(result, "done")
        second_request = json.dumps(seen_messages[1], ensure_ascii=False)
        self.assertIn("context clipped", second_request)
        self.assertNotIn("x" * 1000, second_request)
        self.assertLess(len(second_request), 6500)
