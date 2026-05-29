import json

from madharness_mini.context import ContextFragment, ContextManager

from tests.helpers import HarnessTestCase


def tool_call(call_id="call_1", name="demo"):
    return {
        "id": call_id,
        "function": {"name": name, "arguments": "{}"},
    }


class ContextManagerTests(HarnessTestCase):
    def test_record_assistant_strips_vendor_fields_from_history(self):
        ctx = ContextManager("task", max_tokens=20000)
        ctx.record_assistant(
            {
                "role": "assistant",
                "content": "I will call a tool.",
                "reasoning": "secret provider reasoning",
                "reasoning_details": [{"text": "very long chain"}],
                "refusal": None,
                "tool_calls": [
                    {
                        "id": "call_extra",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "demo",
                            "arguments": {"path": "README.md"},
                            "extra": "drop me",
                        },
                        "provider_field": "drop me too",
                    }
                ],
            }
        )

        rendered = json.dumps(ctx.messages(), ensure_ascii=False)

        self.assertNotIn("reasoning", rendered)
        self.assertNotIn("refusal", rendered)
        self.assertNotIn("provider_field", rendered)
        self.assertIn('"tool_calls"', rendered)
        self.assertIn('\\"path\\": \\"README.md\\"', rendered)

    def test_record_assistant_clips_large_content(self):
        ctx = ContextManager("task", max_tokens=20000)
        ctx.record_assistant({"role": "assistant", "content": "x" * 20000})

        rendered = json.dumps(ctx.messages(), ensure_ascii=False)

        self.assertIn("context clipped", rendered)
        self.assertLess(len(rendered), 12000)

    def test_locked_fragments_and_task_survive_small_budget(self):
        ctx = ContextManager("do the task", max_tokens=90, keep_recent_turns=0)
        ctx.add_fragment(
            ContextFragment(
                id="system",
                source="test",
                text="system rules stay visible",
                priority=0,
            )
        )
        for index in range(5):
            ctx.record_assistant(
                {"role": "assistant", "content": f"old answer {index} " + "x" * 100}
            )

        messages = ctx.messages()

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("system rules stay visible", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "do the task"})
        self.assertTrue(ctx.stats()["truncated"])
        self.assertGreater(ctx.stats()["dropped_entries"], 0)

    def test_tool_turn_is_removed_atomically(self):
        ctx = ContextManager("task", max_tokens=100, keep_recent_turns=1)
        ctx.add_fragment(ContextFragment("system", "test", "system"))
        old_call = tool_call("old_call")
        ctx.record_assistant({"role": "assistant", "content": None, "tool_calls": [old_call]})
        ctx.record_tool_result(
            old_call,
            {
                "ok": True,
                "tool": "demo",
                "summary": "old result",
                "stdout": "old unique output " + "x" * 1000,
            },
        )
        ctx.record_assistant({"role": "assistant", "content": "new answer"})

        rendered = json.dumps(ctx.messages(), ensure_ascii=False)

        self.assertNotIn("old_call", rendered)
        self.assertNotIn("old unique output", rendered)
        self.assertIn("new answer", rendered)

    def test_followup_image_is_not_stored_inside_tool_observation(self):
        ctx = ContextManager("inspect", max_tokens=20000)
        call = tool_call("image_call", "read_image")
        followup = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image from read_image is attached"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        ctx.record_assistant({"role": "assistant", "content": None, "tool_calls": [call]})
        ctx.record_tool_result(
            call,
            {
                "ok": True,
                "tool": "read_image",
                "summary": "read image metadata",
                "attached": True,
            },
            followup,
        )

        messages = ctx.messages()
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        image_messages = [
            item
            for item in messages
            if item.get("role") == "user" and isinstance(item.get("content"), list)
        ]

        self.assertEqual(len(tool_messages), 1)
        self.assertNotIn("data:image", tool_messages[0]["content"])
        self.assertEqual(len(image_messages), 1)
        self.assertIn("data:image/png;base64,abc", json.dumps(image_messages[0]))

    def test_stats_reports_context_size_and_truncation(self):
        ctx = ContextManager("task", max_tokens=100, keep_recent_turns=0)
        ctx.add_fragment(ContextFragment("system", "test", "system"))
        ctx.record_assistant({"role": "assistant", "content": "x" * 1000})

        ctx.messages()
        stats = ctx.stats()
        report = ctx.report()

        self.assertIsInstance(stats["context_tokens_estimate"], int)
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["history_entries"], 1)
        self.assertEqual(
            report["request_tokens_estimate"],
            stats["context_tokens_estimate"],
        )
        self.assertEqual(report["history"]["total_entries"], 1)
        self.assertEqual(len(report["history"]["dropped_entries"]), 1)

    def test_hard_budget_can_drop_recent_history(self):
        ctx = ContextManager("task", max_tokens=130, keep_recent_turns=3)
        ctx.add_fragment(ContextFragment("system", "test", "system"))
        for index in range(3):
            ctx.record_assistant(
                {
                    "role": "assistant",
                    "content": f"recent {index} " + "x" * 1000,
                }
            )

        messages = ctx.messages()
        report = ctx.report()

        self.assertTrue(report["truncated"])
        self.assertTrue(
            any(item.get("forced") for item in report["history"]["dropped_entries"])
        )
        self.assertLessEqual(report["request_tokens_estimate"], report["max_tokens"])
        self.assertNotIn("recent 0", json.dumps(messages, ensure_ascii=False))

    def test_report_describes_fragments_and_tool_clipping_without_content(self):
        ctx = ContextManager("task", max_tokens=400, keep_recent_turns=3)
        ctx.add_fragment(
            ContextFragment(
                id="system",
                source="test-system",
                text="system rules",
                priority=0,
            )
        )
        call = tool_call("call_clip", "run_shell")
        ctx.record_assistant({"role": "assistant", "content": None, "tool_calls": [call]})
        ctx.record_tool_result(
            call,
            {
                "ok": True,
                "tool": "run_shell",
                "summary": "ran command",
                "stdout": "x" * 2000,
            },
        )

        ctx.messages()
        report = ctx.report()
        rendered = json.dumps(report, ensure_ascii=False)

        self.assertEqual(report["fragments"][0]["id"], "system")
        self.assertEqual(report["fragments"][0]["chars"], len("system rules"))
        self.assertEqual(report["history"]["total_entries"], 1)
        self.assertEqual(report["history"]["rendered_entries"], 1)
        self.assertEqual(
            report["history"]["clipped_tool_messages"][0]["tool_call_id"],
            "call_clip",
        )
        self.assertNotIn("x" * 100, rendered)

    def test_report_counts_tool_schemas_in_request_budget(self):
        ctx = ContextManager("task", max_tokens=40)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "large_tool",
                    "description": "tool schema " + "x" * 500,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "context budget exceeded"):
            ctx.messages(tools)
        report = ctx.report()

        self.assertGreater(report["tools_tokens_estimate"], 0)
        self.assertGreater(
            report["request_tokens_estimate"],
            report["messages_tokens_estimate"],
        )
        self.assertTrue(report["over_budget"])
        self.assertTrue(report["hard_limit_exceeded"])
