from madharness_mini.policy import Policy
from madharness_mini.utils import fail, ok, parse_tool_args

from tests.helpers import HarnessTestCase


class PolicyUtilsTests(HarnessTestCase):
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
