"""Загрузка и сохранение локальной конфигурации харнесса."""

import json
import os
from pathlib import Path

from .utils import DEFAULT_CONFIG, STATE_DIR

IMAGE_DETAIL_VALUES = {"auto", "low", "high", "original"}


class Config:
    """Настройки одного запуска CLI: модель, ключ, границы workspace.

    Слои (позже перекрывают раньше): значения по умолчанию,
    `.madharness-mini/config.json`, `.env`, переменные MADHARNESS_MINI_*.
    Поле `root` — абсолютный каталог, внутри которого агент может трогать файлы.
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
        """Создаём каталог состояния и пустой config.json, если их ещё нет.

        Вызывается перед записью трассы, чтобы `traces/` всегда существовал.
        """

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
        """Команда `init`: записываем config.json и возвращаем список изменений.

        Переданные model/base_url/api_key попадают в файл; в списке changes —
        имена полей, которые реально поменялись, плюс `created` для нового файла.
        """

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
        """Подмешиваем в self.data значения из `.env` и окружения процесса.

        Простые строковые поля берём как есть, а vision-настройки приводим
        к типам конфига: bool, int и ограниченный набор detail-режимов.
        """

        env = read_env_file(self.cwd / ".env")
        env.update(
            {k: v for k, v in os.environ.items() if k.startswith("MADHARNESS_MINI_")}
        )
        for field in ("model", "base_url", "api_key"):
            key = f"MADHARNESS_MINI_{field.upper()}"
            if env.get(key):
                self.data[field] = env[key]
        if env.get("MADHARNESS_MINI_SUPPORTS_IMAGE_INPUT"):
            self.data["supports_image_input"] = parse_bool_env(
                "MADHARNESS_MINI_SUPPORTS_IMAGE_INPUT",
                env["MADHARNESS_MINI_SUPPORTS_IMAGE_INPUT"],
            )
        if env.get("MADHARNESS_MINI_MAX_IMAGE_BYTES"):
            self.data["max_image_bytes"] = parse_int_env(
                "MADHARNESS_MINI_MAX_IMAGE_BYTES",
                env["MADHARNESS_MINI_MAX_IMAGE_BYTES"],
            )
        if env.get("MADHARNESS_MINI_IMAGE_DETAIL"):
            detail = env["MADHARNESS_MINI_IMAGE_DETAIL"].strip()
            if detail not in IMAGE_DETAIL_VALUES:
                allowed = ", ".join(sorted(IMAGE_DETAIL_VALUES))
                raise RuntimeError(
                    f"invalid MADHARNESS_MINI_IMAGE_DETAIL: {detail}; allowed: {allowed}"
                )
            self.data["image_detail"] = detail


def read_env_file(path: Path) -> dict[str, str]:
    """Читаем простой `.env`: строки KEY=value, без shell-подстановок.

    Строки с `#` и без `=` пропускаем. Кавычки вокруг значения снимаем.
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


def parse_bool_env(name: str, value: str) -> bool:
    """Читаем булевы настройки из `.env` без неявной магии shell.

    Явные true/false удобнее для учебного конфига: ошибка в значении не прячется
    и не включает vision-режим случайно.
    """

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"invalid {name}: {value}; expected true or false")


def parse_int_env(name: str, value: str) -> int:
    """Читаем целочисленные лимиты из `.env` и отсекаем отрицательные значения."""

    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid {name}: {value}; expected integer") from exc
    if parsed < 0:
        raise RuntimeError(f"invalid {name}: {value}; expected non-negative integer")
    return parsed
