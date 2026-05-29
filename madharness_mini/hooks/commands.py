"""Запуск пользовательских command hooks через subprocess без shell."""

from __future__ import annotations

import json
import os
import subprocess

from ..utils import clipped
from .config import CommandHookConfig
from .types import HookDecision, HookEvent

# Эти переменные нужны локальным рантаймам, но не раскрывают ключи харнесса.
SAFE_INHERITED_ENV = {
    "ComSpec",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SystemRoot",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "WINDIR",
}


class CommandHookProvider:
    """Адаптирует запись hooks.json к общему HookProvider."""

    def __init__(self, config: CommandHookConfig):
        self.config = config
        self.id = config.id

    def matches(self, event: HookEvent) -> bool:
        """Проверяем event name и простой exact-match по data."""

        if event.name != self.config.event:
            return False
        for key, expected in self.config.match.items():
            actual = event.kind if key == "kind" else event.data.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def handle(self, event: HookEvent) -> HookDecision:
        """Передаём событие в stdin и читаем JSON-решение из stdout."""

        payload = json.dumps(event.as_dict(), ensure_ascii=False)
        try:
            proc = subprocess.run(
                [self.config.command, *self.config.args],
                cwd=self.config.cwd,
                env=_hook_env(self.config.env),
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"hook timed out after {self.config.timeout_seconds:g}s"
            ) from exc
        if proc.returncode != 0:
            stderr = clipped((proc.stderr or "").strip(), 1000)
            detail = f"hook exited with code {proc.returncode}"
            if stderr:
                detail = f"{detail}: {stderr}"
            raise RuntimeError(detail)
        return _decision_from_stdout(proc.stdout)


def _decision_from_stdout(stdout: str) -> HookDecision:
    """Пустой stdout означает allow; JSON stdout может вернуть block/message."""

    text = stdout.strip()
    if not text:
        return HookDecision()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid hook stdout JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("invalid hook stdout JSON: expected object")
    ok_value = raw.get("ok", True)
    if not isinstance(ok_value, bool):
        raise RuntimeError("invalid hook stdout JSON: ok must be boolean")
    block = str(raw.get("block") or "").strip()
    message = str(raw.get("message") or "").strip()
    if not ok_value or block:
        return HookDecision(
            ok=False,
            block=block or message or "blocked by hook",
            message=message,
        )
    return HookDecision(ok=True, message=message)


def _hook_env(explicit: dict[str, str]) -> dict[str, str]:
    """Передаём hook только безопасный минимум окружения и явный env."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key in SAFE_INHERITED_ENV and not key.startswith("MADHARNESS_MINI_")
    }
    env.update(explicit)
    return env
