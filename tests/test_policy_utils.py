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
        # Командная подстановка (обратные кавычки) и brace expansion (фигурные
        # скобки вне кавычек) блокируются в любом режиме: почти всегда это либо
        # ошибка, либо попытка инъекции.
        policy = Policy(self.make_cfg())
        allowed, reason = policy.shell_allowed(
            "mkdir -p {css,js/{config,domain},tests}"
        )
        self.assertFalse(allowed)
        self.assertIn("metacharacters", reason)
        self.assertFalse(policy.shell_allowed("echo `whoami`")[0])
        # Обратные кавычки внутри кавычек тоже ловим — это command substitution.
        self.assertFalse(policy.shell_allowed("echo \"`id`\"")[0])

    def test_shell_interpreter_allows_control_operators_by_default(self):
        # По умолчанию allow_shell_interpreter=true: команда идёт через /bin/sh -c,
        # поэтому && ; | и 2>&1 разрешены — это те операции, на которых раньше
        # агент спотыкался (node --version && npm --version, npx vitest run 2>&1).
        policy = Policy(self.make_cfg())
        self.assertTrue(policy.shell_allowed("node --version && npm --version")[0])
        self.assertTrue(policy.shell_allowed("npx vitest run 2>&1")[0])
        self.assertTrue(policy.shell_allowed("git log --oneline | head -5")[0])
        self.assertTrue(policy.shell_allowed("node x; npm y")[0])

    def test_shell_interpreter_off_denies_control_operators(self):
        # Прежний режим без интерпретатора: операторы запрещены, потому что через
        # execve они ушли бы в программу литералом.
        cfg = self.make_cfg()
        cfg.data["allow_shell_interpreter"] = False
        policy = Policy(cfg)
        allowed, reason = policy.shell_allowed("node x && npm y")
        self.assertFalse(allowed)
        self.assertIn("control operators", reason)
        # Обычная команда по-прежнему разрешена.
        self.assertTrue(policy.shell_allowed("node --version")[0])

    def test_shell_denies_redirect_into_files(self):
        # Редирект в произвольный файл — это обход apply_patch/write_file и
        # safe_path. Блокируем любой файловый target.
        policy = Policy(self.make_cfg())
        self.assertFalse(policy.shell_allowed("echo x > src/main.js")[0])
        self.assertFalse(policy.shell_allowed("cat a >> log.txt")[0])
        self.assertFalse(policy.shell_allowed("node x 2> err.log")[0])

    def test_shell_allows_redirect_to_dev_null(self):
        # /dev/null и /dev/std{out,err} нужны для подавления вывода — разрешаем.
        policy = Policy(self.make_cfg())
        self.assertTrue(policy.shell_allowed("npm test 2>/dev/null")[0])
        self.assertTrue(policy.shell_allowed("node x >/dev/null 2>&1")[0])

    def test_shell_interpreter_keeps_denylist(self):
        # Чёрный список разрушительных команд действует и в режиме с интерпретатором.
        policy = Policy(self.make_cfg())
        self.assertFalse(policy.shell_allowed("sudo rm -rf /")[0])
        self.assertFalse(policy.shell_allowed("wget http://x && cat y")[0])

    def test_shell_policy_keeps_literal_metachars(self):
        # Глоббинг и переменные передаются программе буквально: find/grep/awk
        # раскроют их сами. Запрещать их здесь не нужно.
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
