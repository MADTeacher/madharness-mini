"""Работа с локальной конфигурацией `madharness-mini`."""

import json
import os
from pathlib import Path

from .utils import DEFAULT_CONFIG, STATE_DIR


class Config:
    """Конфигурация, с которой выполняется одна команда `madharness-mini`.

    Значения берутся из настроек по умолчанию, локального `config.json`,
    файла `.env` и переменных окружения. Поле `root` задаёт границу файловых
    операций агента.
    """

    def __init__(self, cwd: Path | None = None):
        self.cwd = (cwd or Path.cwd()).resolve()
        self.state_dir = self.cwd / STATE_DIR
        self.data = dict(DEFAULT_CONFIG)
        path = self.state_dir / "config.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.data.update(stored)
            self.data.pop("provider", None)
            self.data.pop("providers", None)
        self.apply_env()
        self.root = (self.cwd / self.data["workspace_root"]).resolve()

    def ensure_dirs(self) -> None:
        """Подготовить каталог состояния и файл конфигурации по умолчанию."""

        (self.state_dir / "traces").mkdir(parents=True, exist_ok=True)
        cfg = self.state_dir / "config.json"
        if not cfg.exists():
            cfg.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def initialize(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> tuple[Path, list[str]]:
        """Создать или обновить `config.json` и перечислить изменённые поля."""

        self.state_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.state_dir / "config.json"
        changes: list[str] = []
        data = dict(self.data)
        updates = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        }
        for key, value in updates.items():
            if value is not None and data.get(key) != value:
                data[key] = value
                changes.append(key)
        if not cfg.exists():
            changes.append("created")
        cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
        (self.state_dir / "traces").mkdir(parents=True, exist_ok=True)
        self.data = data
        self.root = (self.cwd / self.data["workspace_root"]).resolve()
        return cfg, changes

    def apply_env(self) -> None:
        """Применить поддерживаемые переменные из `.env` и окружения процесса."""

        env = read_env_file(self.cwd / ".env")
        env.update(
            {k: v for k, v in os.environ.items() if k.startswith("MADHARNESS_MINI_")}
        )
        for field in ("model", "base_url", "api_key"):
            key = f"MADHARNESS_MINI_{field.upper()}"
            if env.get(key):
                self.data[field] = env[key]


def read_env_file(path: Path) -> dict[str, str]:
    """Прочитать минимальный `.env`-файл с парами `KEY=value`.

    Парсер поддерживает только настройки проекта и не повторяет синтаксис
    командной оболочки полностью.
    """

    if not path.exists():
        return {}
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip("'\"")
    return data
