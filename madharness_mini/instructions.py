"""Сбор системного промпта для модели."""

from importlib import resources


def load_prompt(name: str) -> str:
    """Берём встроенный markdown из madharness_mini/prompts/{name}.md.

    Сейчас используется как минимум `system` — базовые правила агента.
    """

    path = resources.files("madharness_mini").joinpath(
        "prompts",
        f"{name}.md",
    )
    return path.read_text(encoding="utf-8").rstrip()
