"""Загрузка, создание и переопределение настроек харнесса."""

import json
import os
from pathlib import Path

from .utils import DEFAULT_CONFIG, STATE_DIR


class Config:
    """Настройки одного запуска `madharness-mini`.

    Объект собирает значения из настроек по умолчанию, локального
    `.madharness-mini/config.json`, файла `.env` и переменных окружения.
    После инициализации `root` указывает на рабочую папку, внутри которой
    инструменты агента могут читать и менять файлы.
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
        """Создать служебные папки и конфиг, если их ещё нет."""

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
        """Записать начальный `config.json` и вернуть список изменённых полей."""

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
        """Переопределить выбранные настройки из `.env` и окружения процесса."""

        env = read_env_file(self.cwd / ".env")
        env.update(
            {k: v for k, v in os.environ.items() if k.startswith("MADHARNESS_MINI_")}
        )
        for field in ("model", "base_url", "api_key"):
            key = f"MADHARNESS_MINI_{field.upper()}"
            if env.get(key):
                self.data[field] = env[key]


def read_env_file(path: Path) -> dict[str, str]:
    """Прочитать простой `.env` с парами `KEY=value`.

    Парсер намеренно минимальный: он нужен только для локальных настроек
    учебного проекта и не реализует весь синтаксис shell-переменных.
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
