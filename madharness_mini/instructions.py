"""Сбор системного промпта и проектных инструкций."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

AGENTS_MD_MAX_BYTES = 32 * 1024


def load_prompt(name: str) -> str:
    """Загрузить встроенный markdown-промпт из ресурсов пакета."""

    path = resources.files("madharness_mini").joinpath("prompts", f"{name}.md")
    return path.read_text(encoding="utf-8").rstrip()


def clipped_bytes(text: str, limit: int) -> str:
    """Ограничить текст указанным числом байт в кодировке UTF-8."""

    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    prefix = raw[:limit].decode("utf-8", errors="ignore")
    clipped = len(raw) - len(prefix.encode("utf-8"))
    return prefix + f"\n...[clipped {clipped} bytes]"


def load_agents_md(
    root: Path, cwd: Path, max_bytes: int = AGENTS_MD_MAX_BYTES
) -> str:
    """Собрать применимые `AGENTS.md` от общих правил к локальным.

    Сначала учитывается глобальный файл пользователя, затем инструкции из корня
    workspace и вложенных папок по пути к текущей директории. Общий текст
    обрезается по байтовому лимиту перед добавлением в системное сообщение.
    """

    files: list[Path] = []
    home = Path.home() / ".madharness-mini" / "AGENTS.md"
    if home.exists():
        files.append(home)
    try:
        rel = cwd.resolve().relative_to(root)
    except ValueError:
        rel = Path()
    cur = root
    dirs = [root]
    for part in rel.parts:
        cur = cur / part
        dirs.append(cur)
    for item in dirs:
        path = item / "AGENTS.md"
        if path.exists():
            files.append(path)
    chunks = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks.append(f"# Instructions from {path}\n{text}")
    return clipped_bytes("\n\n".join(chunks), max_bytes)
