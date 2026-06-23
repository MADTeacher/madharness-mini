import json
import os
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from madharness_mini.loop import ask, run_agent
from madharness_mini.model import (
    ModelClient,
    ModelRateLimitError,
    ModelTransientError,
    parse_retry_after,
)
from madharness_mini.model_loop import PATCH_RETRY_HINT_THRESHOLD, _maybe_apply_patch_retry_hint
from madharness_mini.trace import Trace, summarize_trace

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

    def test_model_client_wraps_url_errors_as_transient_errors(self):
        cfg = self.make_cfg()
        cfg.data["api_key"] = "token"

        with patch(
            "madharness_mini.model.urllib.request.urlopen",
            side_effect=urllib.error.URLError("tls eof"),
        ):
            with self.assertRaisesRegex(ModelTransientError, "tls eof"):
                ModelClient(cfg).chat([{"role": "user", "content": "hello"}])

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

    def test_ask_retries_transient_model_errors(self):
        cfg = self.make_cfg()
        transient = ModelTransientError("temporary tls eof")
        raw = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch("madharness_mini.loop.ModelClient.chat", side_effect=[transient, raw]),
            patch("madharness_mini.model_loop.time.sleep") as sleep,
        ):
            result, trace_path = ask("hello", cfg)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(1)
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        self.assertIn("model_transient_retry", [event["event"] for event in events])

    def test_ask_traces_error_after_transient_retries_are_exhausted(self):
        cfg = self.make_cfg()
        transient = ModelTransientError("temporary tls eof")

        with (
            patch("madharness_mini.loop.ModelClient.chat", side_effect=transient),
            patch("madharness_mini.model_loop.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "temporary tls eof"):
                ask("hello", cfg)

        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 3])
        traces = sorted((cfg.state_dir / "traces").glob("*.jsonl"))
        self.assertEqual(len(traces), 1)
        events = [
            json.loads(line)
            for line in traces[0].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [event["event"] for event in events if event["event"] == "model_transient_retry"],
            ["model_transient_retry", "model_transient_retry"],
        )
        self.assertTrue(any(event["event"] == "model_error" for event in events))
        self.assertTrue(
            any(
                event["event"] == "session_end"
                and "temporary tls eof" in str(event.get("result"))
                for event in events
            )
        )

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
        finished = next(event for event in events if event["event"] == "model_call_finished")
        self.assertIsInstance(report["request_tokens_estimate"], int)
        self.assertEqual(report["tools_tokens_estimate"], 0)
        self.assertEqual(report["history"]["total_entries"], 0)
        self.assertEqual(started["model_call_id"], finished["model_call_id"])
        self.assertEqual(finished["model_response"]["prompt_tokens"], None)
        summary = summarize_trace(cfg, Path(trace_path).stem)
        self.assertIn("context:", summary)
        self.assertIn("estimated tokens", summary)
        self.assertIn("history: 0/0 entries", summary)

    def test_trace_summary_prefers_exact_id_before_child_prefix(self):
        cfg = self.make_cfg()
        parent = Trace(cfg, "run")
        child = parent.child("subagent", "subagent-planner")
        parent.write("session_end", result="parent result")
        child.write("session_end", result="child result")

        summary = summarize_trace(cfg, parent.id)

        self.assertIn(str(parent.path), summary)
        self.assertIn("parent result", summary)
        self.assertNotIn("child result", summary)

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
        cfg.data["context_max_tokens"] = 8000
        seen_messages = []
        huge_stdout = "x" * 20000

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
        self.assertNotIn("x" * 3000, second_request)
        self.assertLess(len(second_request), 10000)

    def test_apply_patch_retry_hint_after_two_failures(self):
        """Две неудачных apply_patch по пути → во втором observation есть retry_hint."""

        patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch"
        args = {"patch": patch}
        failed = {}
        obs1 = {"ok": False, "tool": "apply_patch", "summary": "expected 1 hunk match, found 0"}
        obs2 = {"ok": False, "tool": "apply_patch", "summary": "expected 1 hunk match, found 0"}

        _maybe_apply_patch_retry_hint("apply_patch", args, obs1, failed)
        # Первая неудача: подсказки ещё нет, счётчик = 1.
        self.assertNotIn("retry_hint", obs1)
        self.assertEqual(failed.get("src/app.py"), 1)

        _maybe_apply_patch_retry_hint("apply_patch", args, obs2, failed)
        # Вторая неудача: порог достигнут, подсказка появилась.
        self.assertIn("retry_hint", obs2)
        self.assertIn("src/app.py", obs2["retry_hint"])

    def test_apply_patch_single_failure_no_hint(self):
        """Одна неудача — без подсказки (порог не достигнут)."""

        args = {"patch": "*** Update File: a.py\n@@\n-old\n+new\n"}
        failed = {}
        obs = {"ok": False, "tool": "apply_patch", "summary": "expected 1 hunk match, found 0"}
        _maybe_apply_patch_retry_hint("apply_patch", args, obs, failed)
        self.assertNotIn("retry_hint", obs)

    def test_apply_patch_success_resets_failure_counter(self):
        """Успешная правка сбрасывает счётчик: после fail, success, fail — без hint."""

        patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old\n+new\n*** End Patch"
        args = {"patch": patch}
        failed = {}
        fail_obs = {"ok": False, "tool": "apply_patch", "summary": "hunk mismatch"}
        success_obs = {"ok": True, "tool": "apply_patch", "summary": "applied patch to 1 file(s)"}

        # fail (count=1), success (сброс → ключ удалён), fail (count=1 снова).
        _maybe_apply_patch_retry_hint("apply_patch", args, fail_obs, failed)
        _maybe_apply_patch_retry_hint("apply_patch", args, success_obs, failed)
        _maybe_apply_patch_retry_hint("apply_patch", args, fail_obs, failed)
        # Порог (2) не достигнут: подсказки нет, счётчик снова = 1 после сброса.
        self.assertNotIn("retry_hint", fail_obs)
        self.assertEqual(failed.get("src/app.py"), 1)

    def test_apply_patch_retry_hint_ignores_other_tools(self):
        """Не-apply_patch инструменты не затрагивают счётчик."""

        failed = {}
        obs = {"ok": False, "tool": "write_file", "summary": "boom"}
        _maybe_apply_patch_retry_hint("write_file", {"path": "a.py"}, obs, failed)
        self.assertEqual(failed, {})
        self.assertNotIn("retry_hint", obs)

    def test_loop_recovers_from_malformed_arguments_with_repair_hint(self):
        """Битый arguments (обрезанная генерация) не валит сессию HTTP 400.

        Модель прислала tool_call с arguments не в формате JSON. Loop пишет
        осмысленный observation с repair_hint и trace-событие tool_call_malformed,
        а на следующем ходу модель может продолжить работу.
        """

        cfg = self.make_cfg()
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
                                        "id": "call_broken",
                                        "function": {
                                            "name": "write_file",
                                            # Незакрытая строка — типичный обрыв генерации.
                                            "arguments": '{"content": "unterminated',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "recovered"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("write something", cfg)

        # Сессия завершилась normally, без падения.
        self.assertEqual(result, "recovered")
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        event_types = [event["event"] for event in events]
        # Новое диагностическое событие появилось.
        self.assertIn("tool_call_malformed", event_types)
        # observation несёт человекочитаемую подсказку модели.
        tool_obs = next(
            event for event in events if event.get("event") == "tool_observation"
        )
        self.assertIn("valid JSON", tool_obs["observation"]["summary"])
        self.assertIn("repair_hint", tool_obs["observation"])
        # Критично: в повторный запрос к модели arguments должен быть валидным '{}'.
        second_request = json.dumps(seen_messages[1])
        self.assertIn('"arguments": "{}"', second_request)
        self.assertNotIn("unterminated", second_request)

    def test_finish_reason_is_logged_in_model_call_finished(self):
        """finish_reason попадает в trace для observability (обрывы генерации)."""

        cfg = self.make_cfg()
        raw = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=[raw]):
            result, trace_path = ask("hello", cfg)

        self.assertEqual(result, "ok")
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        finished = next(
            event for event in events if event.get("event") == "model_call_finished"
        )
        self.assertEqual(finished["model_response"]["finish_reason"], "stop")

    def test_record_usage_called_after_model_call(self):
        """Реальный usage из ответа модели калибрует оценку токенов контекста.

        Запускаем агентский цикл из двух ходов: первый возвращает tool_call с
        большим prompt_tokens в usage, второй — финальный ответ. Калибровка из
        первого хода должна поднять token_ratio во втором (видно в context_report).
        """

        cfg = self.make_cfg()
        call_count = {"n": 0}

        def fake_chat(messages, tools=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "list_files",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 1,
                        "total_tokens": 1_000_001,
                    },
                }
            return {"choices": [{"message": {"content": "done"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("do work", cfg)

        self.assertEqual(result, "done")
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        started_events = [
            event for event in events if event.get("event") == "model_call_started"
        ]
        # Второй model_call должен увидеть откалиброванный ratio > 1.0: провайдер
        # сообщил аномально много токенов относительно нашей эвристики.
        self.assertGreaterEqual(len(started_events), 2)
        second_ratio = started_events[1]["context_report"].get("token_ratio", 1.0)
        self.assertGreater(second_ratio, 1.0)

    def test_loop_handles_real_truncated_arguments_from_flappy3_trace(self):
        """Регрессия на реальном кейсе: обрезанный write_file из flappy3.

        Воспроизводит баг с HTTP 400 от Alibaba: модель зашла в repetition loop,
        упёрлась в лимит токенов, arguments остался незакрытым. Сессия не должна
        падать, а модель должна получить ремонтную подсказку.
        """

        cfg = self.make_cfg()
        # Реальный обрезанный фрагмент из трассы flappy3 (turn 11, write_file).
        truncated_args = (
            '{"content": "/**\\n * @fileoverview GameHub — экран выбора игр.'
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
                                        "id": "call_real",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": truncated_args,
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "recovered after truncation"}}]}

        with patch("madharness_mini.loop.ModelClient.chat", side_effect=fake_chat):
            result, trace_path = run_agent("build gamehub", cfg)

        # Сессия восстановилась, без HTTP 400.
        self.assertEqual(result, "recovered after truncation")
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
        ]
        event_types = [event["event"] for event in events]
        self.assertIn("tool_call_malformed", event_types)
        # В повторный запрос провайдер не ушёл обрезанный фрагмент.
        second_request = json.dumps(seen_messages[1])
        self.assertNotIn("GameHub", second_request)
        self.assertIn('"arguments": "{}"', second_request)

