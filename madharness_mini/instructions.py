from __future__ import annotations

from importlib import resources
from pathlib import Path

from .utils import clipped


def load_prompt(name: str) -> str:
    path = resources.files("madharness_mini").joinpath("prompts", f"{name}.md")
    return path.read_text(encoding="utf-8").rstrip()


def load_agents_md(root: Path, cwd: Path) -> str:
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
        text = clipped(path.read_text(encoding="utf-8"), 6000)
        chunks.append(f"# Instructions from {path}\n{text}")
    return "\n\n".join(chunks)
