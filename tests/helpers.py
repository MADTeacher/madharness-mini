import json
import tempfile
import unittest
from pathlib import Path

from madharness_mini.config import Config


class HarnessTestCase(unittest.TestCase):
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

