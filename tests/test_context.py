import json

from madharness_mini.context import ContextFragment, ContextManager

from tests.helpers import HarnessTestCase


def tool_call(call_id="call_1", name="demo"):
    return {
        "id": call_id,
        "function": {"name": name, "arguments": "{}"},
    }


class ContextManagerTests(HarnessTestCase):
    def test_locked_fragments_and_task_survive_small_budget(self):
        ctx = ContextManager("do the task", max_chars=180, keep_recent_turns=0)
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
        ctx = ContextManager("task", max_chars=220, keep_recent_turns=1)
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
        ctx = ContextManager("inspect", max_chars=20000)
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
        ctx = ContextManager("task", max_chars=260, keep_recent_turns=0)
        ctx.add_fragment(ContextFragment("system", "test", "system"))
        ctx.record_assistant({"role": "assistant", "content": "x" * 1000})

        ctx.messages()
        stats = ctx.stats()

        self.assertIsInstance(stats["context_chars"], int)
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["history_entries"], 1)
