"""Сбор системного промпта."""

from __future__ import annotations

from importlib import resources


def load_prompt(name: str) -> str:
    """Загрузить встроенный markdown-промпт из ресурсов пакета."""

    path = resources.files("madharness_mini").joinpath("prompts", f"{name}.md")
    return path.read_text(encoding="utf-8").rstrip()
