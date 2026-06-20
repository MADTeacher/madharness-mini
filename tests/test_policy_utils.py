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

    def test_shell_policy_denies_shell_only_metachars(self):
        # Фигурные скобки раскрывает только bash: без интерпретатора mkdir
        # создал бы каталог с именем из нераскрытых скобок. Блокируем их.
        policy = Policy(self.make_cfg())
        allowed, reason = policy.shell_allowed(
            "mkdir -p {css,js/{config,domain},tests}"
        )
        self.assertFalse(allowed)
        self.assertIn("without a shell", reason)
        self.assertFalse(policy.shell_allowed("echo `whoami`")[0])
        # Обратные кавычки внутри кавычек тоже ловим — это command substitution.
        self.assertFalse(policy.shell_allowed("echo \"`id`\"")[0])

    def test_shell_policy_keeps_literal_metachars(self):
        # Глоббинг и переменные передаются программе буквально через execve:
        # find/grep/awk раскроют их сами. Запрещать их здесь не нужно.
        policy = Policy(self.make_cfg())
        self.assertTrue(policy.shell_allowed("find . -name '*.py'")[0])
        self.assertTrue(policy.shell_allowed("ls *.txt")[0])
        self.assertTrue(policy.shell_allowed("awk '{print $1}' file.log")[0])
        self.assertTrue(policy.shell_allowed("git log --grep=feat")[0])

    def test_observation_format(self):
        self.assertEqual(ok("x", "done")["ok"], True)
        self.assertEqual(fail("x", "bad")["ok"], False)

    def test_parse_tool_args(self):
        name, args = parse_tool_args(
            {"function": {"name": "read_file", "arguments": '{"path":"hello.txt"}'}}
        )
        self.assertEqual(name, "read_file")
        self.assertEqual(args["path"], "hello.txt")
